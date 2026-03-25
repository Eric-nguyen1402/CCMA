"""
Conformal Cross-Modal Acquisition (CCMA) for Active Learning.

Assumes:
- self.features: (N, D) L2-normalized *student* features used by the classifier.
- self.clip_image_embeds: (N, D) L2-normalized CLIP image embeddings (teacher space).
- self.text_embeds: (C, D) L2-normalized CLIP text embeddings (teacher prototypes).
- self.model.get_probs(X): -> (N, C) probabilities for the student classifier.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple
import os
import logging
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from dotenv import dotenv_values
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from collections import Counter
import csv, datetime, pathlib
import time, math

from ALFM.src.clustering.kmeans import cluster_features, cluster_and_select_gpu
from ALFM.src.query_strategies.base_query import BaseQuery
from ALFM.warm_starts import (
    d2ds_warm_start,
    tcfl_warm_start,
    teacher_quality_gate,
    kmeanspp_warm_start_clip,
)

logger = logging.getLogger(__name__)

def joint_entropy(probs: torch.Tensor) -> torch.Tensor:
    return -(probs * torch.log(probs.clamp_min(1e-9))).sum(dim=1)

@staticmethod
def _js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    p, q: (..., K) distributions (already normalized).
    Returns JS(p || q) per instance in the leading dimensions.
    """
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (torch.log(p) - torch.log(m))).sum(dim=-1)
    kl_qm = (q * (torch.log(q) - torch.log(m))).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)

# --- small utility: ensure float tensor on device and L2-normalized ---
def _as_normalized_float(X):
    """
    Accepts numpy array or torch tensor (N, D).
    Returns: torch.FloatTensor (N, D) on best device, L2-normalized.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(X, np.ndarray):
        feats = torch.from_numpy(X).float().to(device)
    else:
        feats = X.detach().float().to(device)
    feats = F.normalize(feats, dim=1)
    return feats

def _greedy_kcenter(X_np: np.ndarray, k: int) -> np.ndarray:
    """
    Greedy k-center in cosine distance. X_np must be L2-normalized numpy (N, D).
    Returns indices into X_np of length k (or N if N<k).
    """
    N = X_np.shape[0]
    if N <= k:
        return np.arange(N, dtype=np.int64)
    # start with a random point (or 0); random is fine for diversity
    sel = [np.random.randint(N)]
    # cosine distance = 1 - dot
    dmin = 1.0 - (X_np @ X_np[sel[0]])
    for _ in range(1, k):
        idx = int(np.argmax(dmin))
        sel.append(idx)
        dnew = 1.0 - (X_np @ X_np[idx])
        dmin = np.minimum(dmin, dnew)
    return np.array(sel, dtype=np.int64)

def _sklearn_kmeans_select(X_np: np.ndarray, k: int, score_np: np.ndarray | None = None,
                           random_state: int = 42) -> np.ndarray:
    """
    Cluster X_np (N,D) into k clusters with sklearn KMeans and pick ONE rep per cluster.
    If score_np is provided (length N), pick highest-scoring item in each cluster;
    otherwise pick the point closest to cluster centroid.
    Returns indices into X_np (length <= k; fewer if some clusters empty).
    """
    k = max(1, min(k, X_np.shape[0]))
    km = KMeans(n_clusters=k, n_init=10, max_iter=100, random_state=random_state)
    labels = km.fit_predict(X_np)
    reps = []
    for c in range(k):
        idx_c = np.where(labels == c)[0]
        if idx_c.size == 0:
            continue
        if score_np is not None:
            best_local = idx_c[np.argmax(score_np[idx_c])]
        else:
            # pick nearest to centroid
            centroid = km.cluster_centers_[c]
            # cosine distance is fine (features normalized)
            d = 1.0 - (X_np[idx_c] @ centroid)
            best_local = idx_c[np.argmin(d)]
        reps.append(best_local)
    return np.array(reps, dtype=np.int64)

def _faiss_assign_select(feats_tensor: torch.Tensor, k: int):
    """
    Wrap your FAISS GPU path. Expects L2-normalized torch.FloatTensor on device.
    Returns:
        sel_idx: torch.LongTensor indices into feats_tensor (length k' <= k)
        k_used:  int (clusters actually used)
    """
    # You already have this function in your codebase:
    #   cluster_and_select_gpu(tensor, num_clusters, num_samples_to_select)
    sel_idx, k_used = cluster_and_select_gpu(
        feats_tensor,
        num_clusters=k,
        num_samples_to_select=k,
    )
    return sel_idx, k_used


class GPUTimer:
    """GPU-safe timer: uses perf_counter and torch.cuda.synchronize() when available."""
    def __init__(self, sync=True):
        self.sync = sync and torch.cuda.is_available()
        self.t0 = None
    def __enter__(self):
        if self.sync: torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, exc_type, exc, tb):
        if self.sync: torch.cuda.synchronize()
        self.dt = time.perf_counter() - self.t0

def _ensure_csv(path, header):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)

def _append_csv(path, row):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)

def per_k(x, denom, k=1000):
    """Amortized time per-k items; returns NaN if denom==0."""
    return float("nan") if denom == 0 else x * (k / denom)

class Disagreement(BaseQuery):
    def __init__(
        self,
        subpool_size: int = 10000,
        oversampling: int = 10,
        dataset_name: Optional[str] = None,
        model_name: Optional[str] = None,
        teacher_model_name: Optional[str] = None,
        temperature: float = 1.0,
        calibrate_temperature: bool = False,
        top_per_class: int = 0,
        debug: bool = False,
        target_set_size_student: int = 4,
        target_set_size_teacher: int = 8,
        use_teacher: bool = True,
        subpool_mode: str = "full", # {'full', 'random', 'none'}
        final_diversity: str = "faiss", # {'faiss', 'none', 'kcenter', 'sklearn'}
        random_subpool_size: int = 40000,
        fixed_fullpool_cap: int | None = None,
        **params: Any,
    ) -> None:
        super().__init__(**params)

        self.temperature = float(temperature)
        self.calibrate_temperature = bool(calibrate_temperature)
        self.subpool_size = int(subpool_size)
        self.oversampling = int(oversampling)
        self.top_per_class = int(top_per_class) if top_per_class is not None else 0
        self.debug = bool(debug)
        self.final_diversity = str(final_diversity)
        self.subpool_mode = str(subpool_mode)
        self.random_subpool_size = int(random_subpool_size)
        self.fixed_fullpool_cap = fixed_fullpool_cap
        self.target_set_size_student = int(target_set_size_student)
        self.target_set_size_teacher = int(target_set_size_teacher)

        self.dataset_name = dataset_name
        self.model_name = model_name
        self.teacher_model_name = teacher_model_name

        self.use_teacher = bool(use_teacher)
        if not self.use_teacher:
            self.teacher_model_name = None
            logger.info("[Disagreement] Teacher DISABLE by Configuration")

        self._did_warm_start = False
        self.class_perm = None

        # Projection student -> CLIP space
        self.proj_stu2clip: Optional[torch.Tensor] = None  # (D_student, D_clip)
        self._stu2clip_info: Optional[dict] = None

        # Conformal thresholds
        self.q_image: Optional[float] = None
        self.q_text: Optional[float] = None

        # Teacher assets
        self.text_embeds: Optional[torch.Tensor] = None
        self.clip_image_embeds: Optional[np.ndarray] = None

        # Load cached features
        env = dotenv_values()
        feature_root = os.getenv("FEATURE_CACHE_DIR") or env.get("FEATURE_CACHE_DIR")
        if feature_root and dataset_name and self.teacher_model_name:
            teacher_vector_file = os.path.join(
                feature_root, dataset_name, f"{self.teacher_model_name}.hdf"
            )
            logger.info(
                f"[Disagreement] Attempting to load TEACHER assets from: {teacher_vector_file}"
            )
            if os.path.isfile(teacher_vector_file):
                try:
                    with h5py.File(teacher_vector_file, "r") as fh:
                        # Load CLIP image bank from the TEACHER file
                        if "train" in fh and "features" in fh["train"]:
                            img = fh["train/features"][()].astype(np.float32)
                            Xc = F.normalize(torch.from_numpy(img), dim=1)
                            self.clip_image_embeds = Xc.numpy()
                            logger.info(
                                f"[Disagreement] Loaded TEACHER image bank {self.clip_image_embeds.shape} from {teacher_vector_file}"
                            )
                        else:
                            logger.warning(
                                f"[Disagreement] 'train/features' missing in TEACHER file {teacher_vector_file}"
                            )

                        # Load text/label prototypes from the TEACHER file
                        if "text" in fh and "features" in fh["text"]:
                            txt = fh["text/features"][()].astype(np.float32)
                            self.text_embeds = F.normalize(
                                torch.from_numpy(txt), dim=-1
                            )
                            logger.info(
                                f"[Disagreement] Loaded TEACHER text embeds {tuple(self.text_embeds.shape)} from {teacher_vector_file}"
                            )
                        else:
                            logger.warning(
                                f"[Disagreement] 'text/features' missing in TEACHER file {teacher_vector_file}"
                            )
                except Exception as e:
                    logger.warning(
                        f"[Disagreement] Failed reading from TEACHER file {teacher_vector_file}: {e}"
                    )
            else:
                logger.warning(
                    f"[Disagreement] TEACHER feature file not found: {teacher_vector_file}"
                )
        else:
            logger.warning(
                "[Disagreement] Teacher assets not loaded. Missing one of: feature_root, dataset_name, teacher_model_name."
            )
        # After loading teacher assets (so self.clip_image_embeds and self.text_embeds are ready)
        # if (self.subpool_size is None or self.subpool_size <= 0) or (self.oversampling is None or self.oversampling <= 0):
        #     try:
        #         N = self.clip_image_embeds.shape[0] if self.clip_image_embeds is not None else len(self.features)
        #         C = self.text_embeds.shape[0] if self.text_embeds is not None else int(getattr(self, "num_classes", 0))
        #         hparams = self.auto_tune_hparams(N, C)
        #         if self.subpool_size is None or self.subpool_size <= 0:
        #             self.subpool_size = hparams["subpool_size"]
        #         if self.oversampling is None or self.oversampling <= 0:
        #             self.oversampling = hparams["oversampling"]
        #         logger.info(f"[AutoTune] subpool_size={self.subpool_size}, oversampling={self.oversampling} (N={N}, C={C})")
        #     except Exception as e:
        #         logger.warning(f"[AutoTune] Failed to infer adaptive hparams: {e}")
        # root_results = os.getenv("RESULTS_DIR", "ALFM/logs/results")
        # ds = str(self.dataset_name or "unknown_dataset")
        # bk = str(self.model_name or "unknown_backbone")
        # out_dir = os.path.join(root_results, ds, bk)
        # pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
        # self.timing_csv = os.path.join(out_dir, f"disagreement-V5-timing.csv")
        # self.train_timing_csv = os.path.join(out_dir, f"disagreement-V1-train_timing.csv")

        # # create headers ONCE here (not in query)
        # _ensure_csv(
        #     self.timing_csv,
        #     [
        #         "iteration","labeled","unlabeled","num_samples",
        #         "subpool_mode","final_diversity",
        #         "pool_size","topk","picked",
        #         "t_subpool","t_modalities","t_sets","t_score","t_topk","t_diversity",
        #         "t_modalities_per1k","t_sets_per1k","t_score_per1k","t_topk_per1k",
        #         "t_diversity_per1k_sel"
        #     ],
        # )
        # _ensure_csv(self.train_timing_csv, ["iteration","t_train"])

    def _append_diag_row(self, row: dict) -> None:
        try:
            with open(self._diag_csv, "a", newline="") as fh:
                csv.writer(fh).writerow([
                    row.get("timestamp",""),
                    int(row.get("iteration", -1)),
                    int(row.get("num_samples", 0)),
                    int(row.get("labeled", 0)),
                    int(row.get("unlabeled", 0)),
                    float(row.get("student_top1_mean", float("nan"))),
                    float(row.get("student_entropy_mean", float("nan"))),
                    float(row.get("teacher_top1_mean", float("nan"))),
                    float(row.get("teacher_entropy_mean", float("nan"))),
                    float(row.get("frac_disagree_top1", float("nan"))),
                    float(row.get("js_full_mean", float("nan"))),
                    float(row.get("CCMA_GI_mean", float("nan"))),
                    float(row.get("CCMA_GT_mean", float("nan"))),
                    float(row.get("CCMA_overlap_mean", float("nan"))),
                    float(row.get("CCMA_symdiff_mean", float("nan"))),
                    float(row.get("CCMA_identical_pct", float("nan"))),
                    float(getattr(self, "temperature", float("nan"))),
                    int(getattr(self, "target_set_size_student", 0)),
                    int(getattr(self, "target_set_size_teacher", 0)),
                    float(getattr(self, "lam", 0.0)),
                    int(getattr(self, "subpool_size", 0)),
                    int(getattr(self, "oversampling", 0)),
                ])
        except Exception as e:
            logger.warning(f"[DiagCSV] Failed appending diagnostics: {e}")

    def auto_tune_hparams(self,N: int, C: int) -> dict:
        """
        Auto-tune subpool_size and oversampling for active learning.
        Works across small, medium, and large datasets.
        
        Args:
            N: total dataset size
            C: number of classes
        """
        # --- Subpool size ---
        # Rule: ~10x #classes, but also cover diversity for large datasets
        subpool_size = min(
            max(1000, 10 * C),   # at least 1k, ideally 10× #classes
            int(0.2 * N),        # but no more than 20% of dataset
            5000                 # global hard cap
        )

        # --- Oversampling ---
        # Rule: more classes → more oversampling needed
        if C <= 10:            # e.g., CIFAR-10, STL-10
            oversampling = 5
        elif C <= 100:         # CIFAR-100, Caltech101
            oversampling = 8
        elif C <= 300:         # Caltech256
            oversampling = 10
        elif C <= 1000:        # FOOD101, DTD
            oversampling = 12
        else:                  # DomainNet, ImageNet-scale
            oversampling = 15

        return {
            "subpool_size": subpool_size,
            "oversampling": oversampling,
        }

    # ---------- Subpool (diverse) ----------
    def _select_subpool(self) -> NDArray[np.int64]:
        """
        Select a diverse subpool of unlabeled samples using clustering.

        Returns:
            NDArray[np.int64]: Array of unlabeled-relative indices for the selected subpool
        """
        # Prefer CLIP space for geometry (better clustering)
        base_feats = self.clip_image_embeds if self.clip_image_embeds is not None else self.features

        unlabeled_indices = np.flatnonzero(~self.labeled_pool)
        num_unlabeled = len(unlabeled_indices)
        print(f"Number of unlabeled samples: {num_unlabeled}")
        k = min(self.subpool_size, num_unlabeled)

        if k <= 0:
            return np.array([], dtype=np.int64)

        # --- OPTIMIZATION 1: cap candidate pool to avoid OOM ---
        CLUSTER_CANDIDATE_CAP = 50_000 
        if num_unlabeled > CLUSTER_CANDIDATE_CAP:
            logger.info(
                f"[_select_subpool] Capping clustering candidates to {CLUSTER_CANDIDATE_CAP} "
                f"from {num_unlabeled} to prevent memory errors."
            )
            indices_into_unlabeled = np.random.choice(num_unlabeled, CLUSTER_CANDIDATE_CAP, replace=False)
            unlabeled_feats_for_clustering = base_feats[unlabeled_indices[indices_into_unlabeled]]
        else:
            indices_into_unlabeled = np.arange(num_unlabeled)
            unlabeled_feats_for_clustering = base_feats[unlabeled_indices]

        # --- OPTIMIZATION 2: smart cluster count ---
        max_reasonable_clusters = min(
            k,                                           # don't exceed requested subpool size
            len(unlabeled_feats_for_clustering) // 3,    # ~5 pts/cluster min, //4 for CIFAR100 and DomainNet-Real // 3 for Food101
            20000,                                        # 30k for CIFAR100 and DomainNet-Real and 20K Food101
            self.subpool_size
        )
        actual_clusters_req = max(1, max_reasonable_clusters)

        logger.info(
            f"[_select_subpool] Clustering {len(unlabeled_feats_for_clustering)} points into "
            f"<= {actual_clusters_req} clusters (requested subpool_size={k})"
        )

        # --- OPTIMIZATION 3: normalize features (cosine-ish geometry for CLIP) ---
        feats_tensor = F.normalize(
            torch.from_numpy(unlabeled_feats_for_clustering).float().to("cuda" if torch.cuda.is_available() else "cpu"),
            dim=1
        )

        centroid_indices_in_subset: Optional[torch.Tensor] = None
        actual_clusters_used: Optional[int] = None

        # --- GPU/CPU FAISS KMeans with robust selection path ---
        try:
            logger.info(
                f"[_select_subpool] Attempting FAISS KMeans: {feats_tensor.shape[0]} points → "
                f"<= {actual_clusters_req} clusters"
            )

            centroid_indices_in_subset, actual_clusters_used = cluster_and_select_gpu(
                feats_tensor,
                num_clusters=actual_clusters_req,
                num_samples_to_select=min(k, actual_clusters_req),
            )
            logger.info(
                f"[_select_subpool] KMeans assign+select successful: picked {len(centroid_indices_in_subset)} "
                f"samples from {actual_clusters_used} clusters"
            )
        except Exception as e:
            logger.warning(f"[_select_subpool] FAISS path failed: {e}. Falling back to CPU sklearn.")
            centroid_indices_in_subset = None

        # --- CPU sklearn fallback (rare) ---
        if centroid_indices_in_subset is None:
            try:
                from sklearn.cluster import KMeans

                feats_cpu = feats_tensor.cpu().numpy()
                cpu_clusters = min(actual_clusters_req, 1000)

                kmeans = KMeans(
                    n_clusters=cpu_clusters,
                    n_init=5,
                    max_iter=50,
                    random_state=42,
                )
                cluster_labels = kmeans.fit_predict(feats_cpu)

                selected_indices = []
                centroids = kmeans.cluster_centers_
                for cluster_id in range(cpu_clusters):
                    mask = (cluster_labels == cluster_id)
                    if not mask.any():
                        continue
                    members = np.where(mask)[0]
                    d = np.linalg.norm(feats_cpu[members] - centroids[cluster_id], axis=1)
                    closest = members[np.argmin(d)]
                    selected_indices.append(closest)

                centroid_indices_in_subset = torch.tensor(selected_indices[:k], dtype=torch.long)
                actual_clusters_used = cpu_clusters
                logger.info(
                    f"[_select_subpool] CPU sklearn successful: selected {len(centroid_indices_in_subset)} samples"
                )
            except Exception as e:
                logger.error(f"[_select_subpool] CPU clustering also failed: {e}. Using random selection.")
                n_select = min(k, len(unlabeled_feats_for_clustering))
                rand_idx = np.random.choice(len(unlabeled_feats_for_clustering), n_select, replace=False)
                centroid_indices_in_subset = torch.tensor(rand_idx, dtype=torch.long)
                actual_clusters_used = int(n_select)

        # --- Map back to unlabeled-relative indices ---
        selected_unlabeled_relative_indices = indices_into_unlabeled[
            centroid_indices_in_subset.cpu().numpy()
        ]

        selected = np.asarray(selected_unlabeled_relative_indices, dtype=np.int64)

        if len(selected) == 0:
            logger.warning("[_select_subpool] No samples selected, using fallback random selection")
            n_select = min(k, len(indices_into_unlabeled))
            pos = np.random.choice(indices_into_unlabeled.size, n_select, replace=False)
            selected = indices_into_unlabeled[pos]

        logger.info(
            f"[_select_subpool] Final selection: {len(selected)} samples for subpool "
            f"(requested: {k}, clusters used: {actual_clusters_used})"
        )
        return selected.astype(np.int64, copy=False)

    # ---------- Debug: set-size stats ----------
    def ccma_set_stats(self, GI: torch.Tensor, GT: Optional[torch.Tensor]) -> None:
        with torch.no_grad():
            sizeI = GI.sum(1).float()

            def qv(t):
                q = torch.quantile(t, torch.tensor([0.1, 0.5, 0.9]))
                return [float(x) for x in q]

            if GT is None:
                logger.info(
                    f"[CCMA] |Γ_I| mean={sizeI.mean():.2f}, med={sizeI.median():.0f}, q10/50/90={qv(sizeI)}; GT=None"
                )
                return

            sizeT = GT.sum(1).float()
            inter = (GI & GT).sum(1).float()
            union = (GI | GT).sum(1).float()
            symdiff = union - inter
            logger.info(
                f"[CCMA] |Γ_I| mean={sizeI.mean():.2f}, med={sizeI.median():.0f}, q10/50/90={qv(sizeI)}"
            )
            logger.info(
                f"[CCMA] |Γ_T| mean={sizeT.mean():.2f}, med={sizeT.median():.0f}, q10/50/90={qv(sizeT)}"
            )
            logger.info(
                f"[CCMA] overlap mean={inter.mean():.2f}, symdiff mean={symdiff.mean():.2f}, %identical={(symdiff==0).float().mean().item():.2%}"
            )

    # ---------- Align teacher rows to dataset label IDs ----------
    @torch.no_grad()
    def _align_text_embeds_to_labels(self) -> dict:
        if self.text_embeds is None:
            return {"ok": False, "reason": "no text_embeds"}

        idx = np.flatnonzero(self.labeled_pool)
        if idx.size == 0:
            return {"ok": False, "reason": "no labeled pool"}

        # Use CLIP image space for alignment
        Xsrc = (
            self.clip_image_embeds if self.clip_image_embeds is not None else self.features
        )[idx]
        X = F.normalize(torch.from_numpy(Xsrc).float(), dim=1)
        y = torch.from_numpy(self.labels[idx]).long().view(-1)
        T = F.normalize(self.text_embeds.float(), dim=1)

        present = torch.unique(y).tolist()
        if len(present) < 5:
            return {"ok": False, "reason": "too few classes for alignment", "present": len(present)}

        protos, cls_ids = [], []
        for c in present:
            m = y == c
            protos.append(X[m].mean(0, keepdim=True))
            cls_ids.append(c)
        P = F.normalize(torch.cat(protos, dim=0), dim=1)
        S = P @ T.T
        r_idx, c_idx = linear_sum_assignment((-S).cpu().numpy())

        C = T.size(0)
        perm = -np.ones(C, dtype=np.int64)
        for ridx, cidx in zip(r_idx, c_idx):
            perm[cls_ids[ridx]] = cidx

        remaining_text = [j for j in range(C) if j not in c_idx]
        remaining_classes = [j for j in range(C) if perm[j] < 0]
        if remaining_classes:
            g = X.mean(0, keepdim=True)
            sim = (F.normalize(g, dim=1) @ T.T).squeeze(0).cpu().numpy()
            order = np.argsort(-sim)
            fill = [j for j in order if j in remaining_text][: len(remaining_classes)]
            for cls, tj in zip(remaining_classes, fill):
                perm[cls] = tj

        assert (perm >= 0).all(), "alignment produced incomplete permutation"
        self.text_embeds = self.text_embeds[torch.from_numpy(perm)]
        self.class_perm = perm
        return {"ok": True, "mapped": len(present), "example": perm[:10].tolist()}

    # ---------- Projection student -> teacher space ----------
    @torch.no_grad()
    def _fit_student2clip_projection(
        self,
        max_samples: int = 20000,
        reg: float = 1e-3,
    ) -> dict:
        """
        Solve W = argmin || normalize(Xs) W - normalize(Xc) ||_F^2 + reg * ||W||_F^2
        where Xs are student features (N, Ds) and Xc are CLIP image embeds (N, Dc).
        Returns W in R^{Ds x Dc}. Lightweight and deterministic.
        """
        if self.clip_image_embeds is None or self.features is None:
            return {"ok": False, "reason": "missing_features"}

        Xs_np = np.asarray(self.features, dtype=np.float32)
        Xc_np = np.asarray(self.clip_image_embeds, dtype=np.float32)
        N = min(len(Xs_np), len(Xc_np))
        if N == 0:
            return {"ok": False, "reason": "empty_arrays"}

        # Subsample deterministically for speed
        rng = np.random.default_rng(int(getattr(self, "seed", 0)))
        idx = np.arange(N, dtype=np.int64)
        if N > max_samples:
            idx = rng.choice(N, size=max_samples, replace=False)

        Xs = torch.from_numpy(Xs_np[idx]).float()
        Xc = torch.from_numpy(Xc_np[idx]).float()

        # Row-normalize to focus on geometry (cosine alignment)
        Xs = F.normalize(Xs, dim=1)
        Xc = F.normalize(Xc, dim=1)

        Ds = Xs.shape[1]
        Dc = Xc.shape[1]
        I = torch.eye(Ds, dtype=Xs.dtype, device=Xs.device)

        # Ridge closed form: W = (Xs^T Xs + λI)^{-1} Xs^T Xc
        XtX = Xs.T @ Xs
        XtY = Xs.T @ Xc
        W = torch.linalg.solve(XtX + reg * I, XtY)  # (Ds, Dc)

        self.proj_stu2clip = W.contiguous()
        info = {
            "ok": True,
            "Ds": int(Ds),
            "Dc": int(Dc),
            "N_fit": int(len(idx)),
            "reg": float(reg),
        }
        self._stu2clip_info = info
        logger.info(f"[Proj] Fitted student→CLIP projection: {info}")
        return info

    def _project_student_to_clip(self, Xs: torch.Tensor) -> torch.Tensor:
        """Map student features to CLIP space and re-normalize. Expects Xs in float32. Returns (N, Dc)."""
        if self.proj_stu2clip is None:
            return Xs  # fallback (only used when dims already match)
        Z = Xs @ self.proj_stu2clip  # (N, Dc)
        return F.normalize(Z, dim=1)

    # ---------- Modal posteriors ----------
    def _compute_modalities(
        self, idxs: NDArray[np.int64]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Student probs p_I and teacher probs p_T on the subpool (idxs are unlabeled-relative)."""
        unlabeled_mask = ~self.labeled_pool
        feats_np = self.features[unlabeled_mask][idxs]  # student features to the classifier
        image_probs = self.model.get_probs(feats_np)  # (M, C)

        text_probs = None
        if self.text_embeds is not None:
            if self.clip_image_embeds is None:
                logger.warning("[Teacher] No CLIP image bank; teacher logits will be junk.")
                return image_probs, None

            unlabeled_idx = np.flatnonzero(unlabeled_mask)
            abs_ids = unlabeled_idx[idxs]
            clip_x = F.normalize(
                torch.from_numpy(self.clip_image_embeds[abs_ids]).float(), dim=1
            )
            T = F.normalize(self.text_embeds.float(), dim=1)
            logitsT = (clip_x @ T.T) / max(self.temperature, 1e-3)
            text_probs = F.softmax(logitsT, dim=1)

        return image_probs, text_probs

    # ---------- Lifecycle ----------
    def update_state(
        self,
        iteration: int,
        labeled_pool: NDArray[np.bool_],
        model: Any,
    ) -> None:
        super().update_state(iteration, labeled_pool, model)

        # Keep a handle to the (possibly None) model
        self.model = model

        # Reasonable default for CLIP temperature
        if self.temperature < 0.03:
            logger.info(f"[Disagreement] Temperature {self.temperature} too low, using 0.07")
            self.temperature = 0.07

        # Align teacher to dataset label IDs if we have any labels at all
        if self.text_embeds is not None and np.flatnonzero(self.labeled_pool).size > 0:
            info = self._align_text_embeds_to_labels()
            logger.info(f"[Align] {info}")

        # Fit student->CLIP projection if needed
        if self.clip_image_embeds is not None and self.features is not None:
            Ds = int(self.features.shape[1])
            Dc = int(self.clip_image_embeds.shape[1])
            if Ds != Dc and self.proj_stu2clip is None:
                try:
                    self._fit_student2clip_projection(max_samples=20000, reg=1e-3)
                except Exception as e:
                    logger.warning(f"[Proj] Fitting student→CLIP projection failed: {e}")


        if self.debug:
            # Diagnostics can run without a model; they’ll gracefully skip ZS if needed
            self.run_diagnostics()

        # ---- IMPORTANT: only fit conformal thresholds once a trained model is present
        # and we have enough labeled points for a stable calibration.
        enough_labels = np.flatnonzero(self.labeled_pool).size >= 10
        if (model is None) or (not enough_labels):
            logger.info(
                "[Conformal] Skipping calibration (model=%s, labeled=%d).",
                "None" if model is None else "OK",
                int(np.flatnonzero(self.labeled_pool).size),
            )
            return

        try:
            self._fit_conformal_target_size_student(
                target_size=self.target_set_size_student
            )
            self._fit_conformal_target_size_teacher(
                target_size=self.target_set_size_teacher
            )
        except Exception as e:
            logger.warning(f"[Disagreement] _fit_conformal_target_size skipped: {e}")

    # ---------- Zero-shot debug ----------
    @torch.no_grad()
    def _zeroshot_debug(
        self, X: torch.Tensor, T: torch.Tensor, y: torch.Tensor, name: str
    ) -> dict:
        if y.ndim > 1:
            y = y.view(-1)
        logits_raw = X @ T.T  # raw logits (no temperature)
        logger.info(
            f"[Debug {name}] X.shape={X.shape}, T.shape={T.shape}, y.shape={y.shape}"
        )
        logger.info(
            f"[Debug {name}] logits range=[{logits_raw.min():.3f}, {logits_raw.max():.3f}]"
        )
        acc1_raw = (logits_raw.argmax(1) == y).float().mean().item()
        for i in range(min(3, len(y))):
            logger.info(
                f"[Debug {name}] Sample {i}: true={y[i].item()}, pred={logits_raw[i].argmax().item()}"
            )
        logits_temp = logits_raw / max(self.temperature, 1e-3)
        acc1_temp = (logits_temp.argmax(1) == y).float().mean().item()
        top5 = logits_raw.topk(5, dim=1).indices
        hit5 = (top5 == y[:, None]).any(dim=1).float().mean().item()
        chance = 1.0 / T.size(0)
        logger.info(
            f"[Debug {name}] temp={self.temperature:.3f}, acc_raw={acc1_raw:.3f}, acc_temp={acc1_temp:.3f}"
        )
        return {
            "top1": acc1_raw,
            "top5": hit5,
            "chance": chance,
            "usable": acc1_raw > chance + 0.10,
            "temp_acc": acc1_temp,
        }

    @torch.no_grad()
    def run_diagnostics(self) -> dict:
        summary = {}
        N = len(self.features) if hasattr(self, "features") else None
        D_student = int(self.features.shape[1]) if hasattr(self, "features") else None
        has_clip_bank = getattr(self, "clip_image_embeds", None) is not None
        D_clip = int(self.clip_image_embeds.shape[1]) if has_clip_bank else None
        has_text = self.text_embeds is not None
        C_text = int(self.text_embeds.shape[0]) if has_text else None
        D_text = int(self.text_embeds.shape[1]) if has_text else None
        dims = {
            "N": N,
            "D_student": D_student,
            "has_clip_bank": has_clip_bank,
            "D_clip": D_clip,
            "has_text": has_text,
            "C_text": C_text,
            "D_text": D_text,
        }
        summary["dims"] = dims
        logger.info(f"[Diag] dims: {dims}")

        try:
            unl = np.flatnonzero(~self.labeled_pool)
            pool = self._select_subpool()
            ok_index = bool(pool.ndim == 1 and (pool.size == 0 or pool.max() < len(unl)))
        except Exception:
            ok_index = False
        summary["indexing_ok"] = ok_index
        logger.info(
            f"[Diag] subpool indexing_ok={ok_index} (subpool_size={int(pool.size) if 'pool' in locals() else 'n/a'})"
        )

        idx = (
            np.flatnonzero(self.labeled_pool) if hasattr(self, "labeled_pool") else np.array([], dtype=int)
        )
        if idx.size == 0:
            msg = {"usable": False, "reason": "no labeled pool"}
            summary["zeroshot_clip"] = msg
            summary["zeroshot_student"] = msg
            logger.warning("[Diag] No labeled pool; skipping zero-shot.")
            return summary

        y = torch.from_numpy(self.labels[idx]).long().view(-1) if hasattr(self, "labels") else None
        if y is None:
            msg = {"usable": False, "reason": "labels missing"}
            summary["zeroshot_clip"] = msg
            summary["zeroshot_student"] = msg
            logger.warning("[Diag] labels missing; skipping zero-shot.")
            return summary

        T = F.normalize(self.text_embeds.float(), dim=1) if has_text else None
        Xc = None
        if has_clip_bank:
            Xc = F.normalize(
                torch.from_numpy(self.clip_image_embeds[idx]).float(), dim=1
            )
        Xs = None
        if hasattr(self, "features") and self.features is not None:
            Xs = F.normalize(torch.from_numpy(self.features[idx]).float(), dim=1)

        cos = None
        if Xc is not None and Xs is not None:
            Xs_cmp = Xs
            if Xs_cmp.shape[1] != Xc.shape[1]:
                Xs_cmp = self._project_student_to_clip(Xs_cmp)
            cos = torch.sum(Xc * Xs_cmp, dim=1).mean().item()
            logger.info(f"[Diag] mean cos(student→CLIP vs CLIP-img): {cos:.3f}")
        summary["mean_cos_student_vs_clip"] = cos

        if T is not None and Xc is not None:
            summary["zeroshot_clip"] = self._zeroshot_debug(Xc, T, y, "CLIP")
        else:
            summary["zeroshot_clip"] = {"usable": False, "reason": "missing T or Xc"}
            logger.warning("[Diag] Skipping ZS_clip (missing T or Xc).")

        if T is not None and Xs is not None:
            # If dims mismatch, project student to CLIP space
            if Xs.shape[1] != T.shape[1]:
                Xs = self._project_student_to_clip(Xs)
            summary["zeroshot_student"] = self._zeroshot_debug(Xs, T, y, "student")
        else:
            summary["zeroshot_student"] = {"usable": False, "reason": "missing T or Xs"}

        return summary

    # ---------- Conformal prediction sets ----------
    @torch.no_grad()
    def _solve_q_for_target(self, a: torch.Tensor, target_size: int, max_iters: int = 30) -> Tuple[float, float]:
        """
        Solve for q such that avg_row(|{c: a_ic <= q}|) ~= target_size, using bisection on the *current* batch.
        a: (M, C) nonconformity = -log p, >= 0
        Returns (q_star, achieved_avg_size).
        """
        assert a.ndim == 2
        M, C = a.shape
        tgt = float(max(1, min(int(target_size), C)))

        # Low: ensures at least one class per row (max of each row's min).
        amin_row = a.min(dim=1).values
        low = float(amin_row.max().item())

        # High: includes all classes.
        high = float(a.max().item())

        def avg_size(q: float) -> float:
            return (a <= q).sum(dim=1).float().mean().item()

        size_low = avg_size(low)
        size_high = avg_size(high)

        # Clip if target is out of bounds (should be rare)
        if tgt <= size_low + 1e-6:
            q_star = low
            return q_star, size_low
        if tgt >= size_high - 1e-6:
            q_star = high
            return q_star, size_high

        l, h = low, high
        for _ in range(max_iters):
            m = 0.5 * (l + h)
            sz = avg_size(m)
            if sz > tgt:
                h = m
            else:
                l = m
        q_star = 0.5 * (l + h)
        achieved = avg_size(q_star)
        return float(q_star), float(achieved)

    def _prediction_sets(
        self,
        image_probs: torch.Tensor,
        text_probs: Optional[torch.Tensor],
        qI_override: Optional[float] = None,   # kept for API compat; not strictly needed now
        qT_override: Optional[float] = None,
        ensure_nonempty: bool = True,
    ):
        """
        Return boolean masks for Γ_I, Γ_T of shape (M, C).

        IMPORTANT: We *re-calibrate* q on the *current subpool* to hit the target set size,
        which fixes the labeled→unlabeled distribution shift you observed.
        """
        eps = 1e-9

        if not isinstance(image_probs, torch.Tensor):
            image_probs = torch.from_numpy(image_probs).float()
        else:
            image_probs = image_probs.float()

        if text_probs is not None and not isinstance(text_probs, torch.Tensor):
            text_probs = torch.from_numpy(text_probs).float()
        if text_probs is not None:
            text_probs = text_probs.float()

        # Clamp + renorm (numerical safety)
        image_probs = image_probs.clamp_min(eps)
        image_probs = image_probs / image_probs.sum(dim=1, keepdim=True).clamp_min(eps)

        if text_probs is not None:
            text_probs = text_probs.clamp_min(eps)
            text_probs = text_probs / text_probs.sum(dim=1, keepdim=True).clamp_min(eps)

        # --- Unsupervised re-solve q_I on the current subpool ---
        aI = -torch.log(image_probs)               # (M, C)
        qI_star, achieved_I = self._solve_q_for_target(
            aI, target_size=getattr(self, "target_set_size_student", 3)
        )
        GI = (aI <= qI_star)

        # --- Unsupervised (optional) re-solve q_T on the current subpool ---
        GT = None
        if text_probs is not None:
            aT = -torch.log(text_probs)            # (M, C)
            qT_star, achieved_T = self._solve_q_for_target(
                aT, target_size=getattr(self, "target_set_size_teacher", 3)
            )
            GT = (aT <= qT_star)

        # --- Safety: ensure non-empty sets (should already be guaranteed by construction)
        if ensure_nonempty:
            # Γ_I
            empty_I = (GI.sum(dim=1) == 0)
            if empty_I.any():
                top1 = image_probs[empty_I].argmax(dim=1)
                GI[empty_I, :] = False
                GI[empty_I, top1] = True

            # Γ_T
            if GT is not None:
                empty_T = (GT.sum(dim=1) == 0)
                if empty_T.any():
                    top1 = text_probs[empty_T].argmax(dim=1)
                    GT[empty_T, :] = False
                    GT[empty_T, top1] = True

        return GI, GT

    def _conformal_score(
        self,
        GI: torch.Tensor,
        GT: Optional[torch.Tensor],
        image_probs: torch.Tensor = None,
        text_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        JS-based disagreement over the union set U = Γ_I ∪ Γ_T.

        score = w_js * JS( p_I|_U  ||  p_T|_U )  +  (1-w_js) * H(p_I)

        - JS captures true cross-modal conflict *where it matters* (the union set).
        - The entropy term captures the student's intrinsic uncertainty.
        - The gate w_js is now a parameter-free ratio of model confidences.
        """
        assert image_probs is not None, "image_probs required"
        if GT is None or text_probs is None:
            # Fallback: just use entropy if teacher isn't available here
            return joint_entropy(image_probs)

        eps = 1e-9
        GI = GI.bool()
        GT = GT.bool()
        U = (GI | GT)                                  # (M, C)

        # Guard against any empty unions (shouldn't happen with ensure_nonempty)
        empty = (U.sum(dim=1) == 0)
        if empty.any():
            # Force top1 of student into U for those rows
            top1 = image_probs[empty].argmax(dim=1)
            U[empty, :] = False
            U[empty, top1] = True

        # Gather and renormalize over the union for each row
        def renorm_over_union(P: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            P_masked = P.masked_fill(~mask, 0.0)
            Z = P_masked.sum(dim=1, keepdim=True).clamp_min(eps)
            return P_masked / Z

        pI_u = renorm_over_union(image_probs, U)       # (M, C), zero outside U
        pT_u = renorm_over_union(text_probs, U)

        js = _js_divergence(pI_u, pT_u)           # (M,)
        entI = joint_entropy(image_probs)               # (M,)

        # Test new score
        confI = image_probs.max(dim=1).values
        confT = text_probs.max(dim=1).values
        w_js  = confT / (confT + confI + eps)  # (M,)

        # Final score
        score = w_js * js + (1.0 - w_js) * entI
        return score

    @torch.no_grad()
    def _fit_conformal_target_size_student(
        self,
        target_size: int = 2,
        subset_size: int = 1000,
        max_iters: int = 30,
        eps: float = 1e-9,
    ) -> dict:
        idx_all = np.flatnonzero(self.labeled_pool)
        if idx_all.size < 10:
            msg = "[Conformal-Student] Too few labeled points for calibration"
            logger.warning(msg)
            return {"ok": False, "reason": "too_few_labels", "n_labeled": int(idx_all.size)}

        m = int(min(idx_all.size, max(50, subset_size)))
        cal_idx = np.random.choice(idx_all, size=m, replace=False)
        feats = self.features[cal_idx]
        probs = self.model.get_probs(feats)
        if not isinstance(probs, torch.Tensor):
            probs = torch.from_numpy(np.asarray(probs))
        probs = probs.float()

        logger.info(f"[Conformal-Student] probs after model: {probs.shape}, min={probs.min():.6f}, max={probs.max():.6f}")
        
        C = probs.shape[1]
        probs = probs.clamp_min(eps)
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(eps)

        logger.info(f"[Conformal-Student] probs after clamp+renorm: min={probs.min():.6f}, max={probs.max():.6f}")
        
        aI = -torch.log(probs)

        logger.info(f"[Conformal-Student] aI stats: min={aI.min().item():.6f}, max={aI.max().item():.6f}, mean={aI.mean().item():.6f}")
        
        tgt = float(np.clip(target_size, 1, C))
        amin_row = aI.min(dim=1).values
        low = float(amin_row.max().item())
        high = float(aI.max().item())

        def avg_set_size_at(q: float) -> float:
            return (aI <= q).sum(dim=1).float().mean().item()

        size_low = avg_set_size_at(low)
        size_high = avg_set_size_at(high)
        if not (1.0 - 1e-6 <= size_low <= C + 1e-6 and abs(size_high - C) < 1e-6):
            logger.warning(
                f"[Conformal-Student] unexpected endpoint sizes: size_low={size_low:.3f}, size_high={size_high:.3f}, C={C}"
            )

        if tgt <= size_low + 1e-6:
            q_star = low
        elif tgt >= size_high - 1e-6:
            q_star = high
        else:
            l, h = low, high
            for _ in range(max_iters):
                mid = 0.5 * (l + h)
                sz = avg_set_size_at(mid)
                if sz > tgt:
                    h = mid
                else:
                    l = mid
            q_star = 0.5 * (l + h)

        self.q_image = float(q_star)
        achieved = avg_set_size_at(self.q_image)
        logger.info(
            f"[Conformal-Student] q_image={self.q_image:.4f} @ target_size={tgt:.2f} (achieved={achieved:.2f}, m={m}, C={C})"
        )
        return {
            "ok": True,
            "q_image": self.q_image,
            "target_size": tgt,
            "achieved_size": achieved,
            "m": m,
            "C": C,
        }

    @torch.no_grad()
    def _fit_conformal_target_size_teacher(
        self,
        target_size: int = 2,
        subset_size: int = 1000,
        max_iters: int = 30,
        eps: float = 1e-9,
    ):
        if self.text_embeds is None or self.clip_image_embeds is None:
            logger.info("[Conformal-Teacher] missing teacher components; skipping")
            return {"ok": False, "reason": "missing_teacher"}

        idx_all = np.flatnonzero(self.labeled_pool)
        if idx_all.size < 10:
            return {"ok": False, "reason": "too_few_labels", "n_labeled": int(idx_all.size)}

        m = int(min(idx_all.size, max(50, subset_size)))
        cal_idx = np.random.choice(idx_all, size=m, replace=False)

        tau = max(self.temperature, 1e-3)
        Xc = F.normalize(
            torch.from_numpy(self.clip_image_embeds[cal_idx]).float(), dim=1
        )
        T = F.normalize(self.text_embeds.float(), dim=1)
        logits = (Xc @ T.T) / tau
        pT = F.softmax(logits, dim=1).float().clamp_min(eps)
        pT = pT / pT.sum(dim=1, keepdim=True).clamp_min(eps)
        aT = -torch.log(pT)

        C = aT.shape[1]
        tgt = float(np.clip(target_size, 1, C))
        amin_row = aT.min(dim=1).values
        low = float(amin_row.max().item())
        high = float(aT.max().item())

        def avg_size(q: float) -> float:
            return (aT <= q).sum(1).float().mean().item()

        if tgt <= avg_size(low) + 1e-6:
            q_star = low
        elif tgt >= avg_size(high) - 1e-6:
            q_star = high
        else:
            l, h = low, high
            for _ in range(max_iters):
                mid = 0.5 * (l + h)
                sz = avg_size(mid)
                if sz > tgt:
                    h = mid
                else:
                    l = mid
            q_star = 0.5 * (l + h)

        self.q_text = float(q_star)
        achieved = avg_size(self.q_text)
        logger.info(
            f"[Conformal-Teacher] q_text={self.q_text:.4f} @ target_size={tgt:.2f} (achieved={achieved:.2f}, m={m}, C={C})"
        )
        return {
            "ok": True,
            "q_text": self.q_text,
            "achieved_size": achieved,
            "target_size": tgt,
            "m": m,
            "C": C,
        }

    def _final_diversity_pick(self, X_local: np.ndarray, k: int, method: str, score_local: np.ndarray | None = None) -> np.ndarray:
        """
        X_local: (M, D) numpy L2-normalized features for the *selected_pool* (local space).
        k:       number of final items to pick.
        method:  {'faiss','sklearn','kcenter','none'}
        score_local: optional scores aligned to X_local (used by sklearn selection).

        Returns indices into X_local (length <= k). Caller maps them back to absolute indices.
        """
        M = X_local.shape[0]
        if M == 0:
            return np.array([], dtype=np.int64)
        k = max(1, min(k, M))

        if method == "none":
            return np.arange(k, dtype=np.int64)

        if method == "kcenter":
            return _greedy_kcenter(X_local, k)

        if method == "sklearn":
            return _sklearn_kmeans_select(X_local, k, score_np=score_local)

        if method == "faiss":
            try:
                feats = _as_normalized_float(X_local)  # torch tensor on device
                sel_idx_t, k_used = _faiss_assign_select(feats, k)
                return sel_idx_t.detach().cpu().numpy()
            except Exception as e:
                logger.warning(f"[diversity=faiss] fallback to sklearn due to: {e}")
                return _sklearn_kmeans_select(X_local, k, score_np=score_local)

        raise ValueError(f"Unknown final_diversity method: {method}")
    
    # ---------- Main query method ----------
    def query(self, num_samples: int) -> NDArray[np.bool_]:
        """Select points to label based on multimodal uncertainty and diversity."""
        mask = np.zeros(len(self.features), dtype=bool)
        unlabeled_idx = np.flatnonzero(~self.labeled_pool)
        if num_samples <= 0 or len(unlabeled_idx) == 0:
            return mask
        # --------------Step 1: select a subpool of unlabeled points to score--------------
        with GPUTimer() as t_sub:
            if self.subpool_mode == "full":
                pool_idxs = self._select_subpool()
            elif self.subpool_mode == "random":
                m = min(len(unlabeled_idx), self.random_subpool_size)
                pool_idxs = np.random.choice(len(unlabeled_idx), m, replace=False)
            elif self.subpool_mode == "none":
                # score everything (optionally cap for practicality)
                m = len(unlabeled_idx) if self.fixed_fullpool_cap is None else min(len(unlabeled_idx), self.fixed_fullpool_cap)
                pool_idxs = np.arange(m, dtype=np.int64)
            else:
                raise ValueError(f"Unknown subpool_mode: {self.subpool_mode}")

            if len(pool_idxs) == 0:
                return mask
            if pool_idxs.max(initial=-1) >= unlabeled_idx.size:
                raise RuntimeError(
                    "[query] _select_subpool returned indices outside unlabeled range."
                )
        pool_size = int(len(pool_idxs))

        # --------------Step 2: Compute modalities on the subpool--------------
        with GPUTimer() as t_mod:
            image_p, text_p = self._compute_modalities(pool_idxs)

        # --------------Step 3: Compute conformal prediction sets--------------
        with GPUTimer() as t_set:
            GI, GT = self._prediction_sets(
                image_p,
                text_p,
                qI_override=self.q_image, 
                qT_override=self.q_text,
                ensure_nonempty=True,
            )

        # --------------Step 4: Conformal Scoring--------------
        with GPUTimer() as t_score:
            score = self._conformal_score(GI, GT, image_probs=image_p, text_probs=text_p)

        # --------------Step 5: Diversity Selection--------------
        # Oversample top candidates
        with GPUTimer() as t_topk:
            topk = int(min(score.shape[0], max(num_samples * self.oversampling, num_samples)))
            top_idxs = torch.topk(score, topk).indices.cpu().numpy()  # indices into pool_idxs
            selected_pool = pool_idxs[top_idxs]  # unlabeled-relative
            topk_ct = int(len(selected_pool))

        # Final diversity selection in CLIP space if available
        X_all = (
            self.clip_image_embeds if self.clip_image_embeds is not None else self.features
        )[~self.labeled_pool]
        X = F.normalize(torch.from_numpy(X_all[selected_pool]).float(), dim=1).numpy()

        # --- Diversity pick via FAISS KMeans (assign+select) with sklearn fallback ---
        with GPUTimer() as t_div:
            k = max(1, num_samples)
            # (optional) local scores aligned with X for sklearn mode
            score_local = score[top_idxs].detach().cpu().numpy() if isinstance(score, torch.Tensor) else score[top_idxs]
            sel_local = self._final_diversity_pick(X, k, method=self.final_diversity, score_local=score_local)
            picked_pool = selected_pool[sel_local]

        # Ensure we have valid results
        if picked_pool is None or len(picked_pool) == 0:
            logger.warning("[query] Both clustering methods failed, falling back to top-k selection")
            picked_pool = selected_pool[:num_samples]
        picked_ct = int(min(len(picked_pool), k))
        final_subpool = picked_pool[: min(len(picked_pool), num_samples)]
        mask[unlabeled_idx[final_subpool]] = True  
        
        # # Append timing row (NO header calls here)
        # labeled_ct = int(np.flatnonzero(self.labeled_pool).size)
        # unlabeled_ct = int((~self.labeled_pool).sum())

        # _append_csv(self.timing_csv, [
        #     int(getattr(self, "iteration", -1)),
        #     labeled_ct, unlabeled_ct, int(num_samples),
        #     str(self.subpool_mode), str(self.final_diversity),
        #     pool_size, topk_ct, picked_ct,
        #     t_sub.dt, t_mod.dt, t_set.dt, t_score.dt, t_topk.dt, t_div.dt,
        #     per_k(t_mod.dt, pool_size), per_k(t_set.dt, pool_size),
        #     per_k(t_score.dt, pool_size), per_k(t_topk.dt, pool_size),
        #     per_k(t_div.dt, picked_ct),
        # ])
        return mask
