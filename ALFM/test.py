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
        diversity: str = "kmeans",
        target_set_size_student: int = 4,
        target_set_size_teacher: int = 8,
        lam: float = 0.2,
        **params: Any,
    ) -> None:
        super().__init__(**params)

        self.temperature = float(temperature)
        self.calibrate_temperature = bool(calibrate_temperature)
        self.subpool_size = int(subpool_size)
        self.oversampling = int(oversampling)
        self.top_per_class = int(top_per_class) if top_per_class is not None else 0
        self.debug = bool(debug)
        self.diversity = str(diversity)
        self.target_set_size_student = int(target_set_size_student)
        self.target_set_size_teacher = int(target_set_size_teacher)

        self.dataset_name = dataset_name
        self.model_name = model_name
        self.teacher_model_name = teacher_model_name

        self._did_warm_start = False
        self.class_perm = None
        # conformal parameters
        self.lam = float(lam)

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
            len(unlabeled_feats_for_clustering) // 5,    # ~5 pts/cluster min
            10000,                                        # a safety cap for memory
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
            selected = np.random.choice(len(indices_into_unlabeled), n_select, replace=False)

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

    # ---------- Warm start techniques ----------
    def _as_absolute_indices(self, picks: np.ndarray) -> np.ndarray:
        """
        Map any index array to absolute dataset indices.
        If picks are already absolute (0 <= idx < N), return as-is.
        If called BEFORE self.labeled_pool exists, we assume absolute and validate.
        If picks are unlabeled-relative, map via the unlabeled index array.
        """
        picks = np.asarray(picks, dtype=np.int64).ravel()
        if picks.size == 0:
            return picks

        N = len(self.features)

        # If labeled_pool does not exist yet, we must treat indices as absolute.
        if not hasattr(self, "labeled_pool"):
            if picks.min() < 0 or picks.max() >= N:
                raise ValueError(
                    f"[Warm-start] Got indices before labeled_pool exists, and they are out of [0, {N})"
                )
            return picks

        unl = np.flatnonzero(~self.labeled_pool)

        # If any index is outside [0, N), assume unlabeled-relative and map
        if picks.min() < 0 or picks.max() >= N:
            if unl.size == 0:
                return np.asarray([], dtype=np.int64)
            if picks.max() >= unl.size or picks.min() < 0:
                raise ValueError(
                    f"[Warm-start] Unlabeled-relative indices out of range: max={picks.max()} vs unl_size={unl.size}"
                )
            return unl[picks]

        # Looks like absolute
        return picks

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
    def _to_probs(self, P: torch.Tensor) -> torch.Tensor:
        if not isinstance(P, torch.Tensor):
            P = torch.from_numpy(np.asarray(P))
        P = P.float()
        eps = 1e-9
        row_sum = P.sum(dim=1, keepdim=True)

        looks_like_probs = (P.min() >= -1e-6) and (P.max() <= 1.0 + 1e-3) \
                        and torch.all(torch.abs(row_sum - 1.0) <= 1e-3)

        if looks_like_probs:
            P = P / row_sum.clamp_min(eps)          # already probs → renorm just in case
        else:
            P = F.softmax(P, dim=1)                 # treat as logits/scores

        return P.clamp_min(eps)

    def _prediction_sets(
        self,
        image_probs: torch.Tensor,
        text_probs: Optional[torch.Tensor],
        qI_override: Optional[float] = None,
        qT_override: Optional[float] = None,
        ensure_nonempty: bool = True,
    ):
        """Return boolean masks for Γ_I, Γ_T of shape (M,C)."""

        eps = 1e-9
        if not isinstance(image_probs, torch.Tensor):
            image_probs = torch.from_numpy(image_probs).float()
        if text_probs is not None and not isinstance(text_probs, torch.Tensor):
            text_probs = torch.from_numpy(text_probs).float()
        
        logger.info(f"[DEBUG _prediction_sets] Input image_probs shape: {image_probs.shape}, range: [{image_probs.min():.6f}, {image_probs.max():.6f}]")

        image_probs = self._calibrate_probabilities(image_probs, method="temperature")

        pI = self._to_probs(image_probs)
        pT = self._to_probs(text_probs) if text_probs is not None else None

        logger.info(f"[DEBUG _prediction_sets] After _to_probs pI range: [{pI.min():.6f}, {pI.max():.6f}]")

        if qI_override is not None:
            qI = qI_override
            logger.info(f"[DEBUG _prediction_sets] Using qI_override: {qI:.6f}")
        elif getattr(self, "q_image", None) is not None:
            qI = self.q_image
            logger.info(f"[Prediction Sets] Using stored q_image={qI:.4f}")
            logger.info(f"[DEBUG _prediction_sets] Using stored q_image: {qI:.6f}")
        else:
            # Fallback: compute empirical quantile to target ~3 classes on average
            aI = -torch.log(pI.clamp_min(eps))
            # Sort all scores and pick quantile that gives target size
            target_size = getattr(self, "target_set_size_student", 3)
            target_fraction = target_size / pI.shape[1]  # target_size / num_classes
            qI = float(torch.quantile(aI.reshape(-1), 1.0 - target_fraction).item())
            logger.info(f"[Prediction Sets] Computed fallback q_image={qI:.4f} for target size {target_size}")
        
        aI = -torch.log(pI.clamp_min(eps))
        logger.info(f"[DEBUG _prediction_sets] Conformal scores aI range: [{aI.min():.6f}, {aI.max():.6f}]")
        
        GI = aI <= qI
        initial_size_I = GI.sum(dim=1).float().mean().item()
        logger.info(f"[DEBUG _prediction_sets] Initial GI average size: {initial_size_I:.2f}")

        GT = None
        if text_probs is not None:
            if qT_override is not None:
                qT = qT_override
            elif getattr(self, "q_text", None) is not None:
                qT = self.q_text
                logger.info(f"[Prediction Sets] Using stored q_text={qT:.4f}")
            else:
                # Fallback: compute empirical quantile
                aT = -torch.log(pT.clamp_min(eps))
                target_size = getattr(self, "target_set_size_teacher", 3)
                target_fraction = target_size / pT.shape[1]
                qT = float(torch.quantile(aT.reshape(-1), 1.0 - target_fraction).item())
                logger.info(f"[Prediction Sets] Computed fallback q_text={qT:.4f} for target size {target_size}")
                
            GT = -torch.log(pT.clamp_min(eps)) <= qT
        
        actual_size_I = GI.sum(dim=1).float().mean().item()
        target_I = getattr(self, "target_set_size_student", 3)
        if actual_size_I > target_I * 2:
            logger.warning(f"[Prediction Sets] GI size {actual_size_I:.2f} is much larger than target {target_I}. "
                       "This indicates a calibration issue.")
        
        if GT is not None:
            actual_size_T = GT.sum(dim=1).float().mean().item()
            target_T = getattr(self, "target_set_size_teacher", 3)
            if actual_size_T > target_T * 2:
                logger.warning(f"[Prediction Sets] GT size {actual_size_T:.2f} is much larger than target {target_T}.")

        # --- Fix 3: Better ensure_nonempty logic ---
        if ensure_nonempty:
            min_size = 1  # At least 1 class per prediction set
            
            # For image sets
            empty_mask_I = GI.sum(dim=1) == 0
            if empty_mask_I.any():
                logger.info(f"[Prediction Sets] Fixing {empty_mask_I.sum()} empty image sets")
                # For empty sets, include the top prediction
                for i in torch.where(empty_mask_I)[0]:
                    top_idx = pI[i].argmax()
                    GI[i, top_idx] = True
            
            # For text sets
            if GT is not None:
                empty_mask_T = GT.sum(dim=1) == 0
                if empty_mask_T.any():
                    logger.info(f"[Prediction Sets] Fixing {empty_mask_T.sum()} empty text sets")
                    for i in torch.where(empty_mask_T)[0]:
                        top_idx = pT[i].argmax()
                        GT[i, top_idx] = True

        # --- Final verification (remove this warning once working) ---
        final_size_I = GI.sum(dim=1).float().mean().item()
        if abs(final_size_I - target_I) > 1.0:
            logger.info(f"[Prediction Sets] Final GI size: {final_size_I:.2f} (target: {target_I})")
        
        if GT is not None:
            final_size_T = GT.sum(dim=1).float().mean().item()
            if abs(final_size_T - target_T) > 1.0:
                logger.info(f"[Prediction Sets] Final GT size: {final_size_T:.2f} (target: {target_T})")

        return GI, GT

    def _conformal_score(
        self,
        GI: torch.Tensor,
        GT: Optional[torch.Tensor],
        lam: float = 0.01,
        image_probs: torch.Tensor = None,
    ) -> torch.Tensor:
        if GT is None:
            # entropy fallback unchanged
            assert image_probs is not None, "image_probs required for entropy fallback"
            return joint_entropy(image_probs)

        inter = (GI & GT).sum(1).float()
        uni = (GI | GT).sum(1).float().clamp_min(1.0)
        sym = uni - inter  # |Δ|
        sizeI = GI.sum(1).float()
        sizeT = GT.sum(1).float()

        # Jaccard-style normalization
        sym_norm = sym / uni
        size_norm = (sizeI + sizeT) / (2.0 * uni)
        logger.info(f"lam={lam}, sym_norm.mean={sym_norm.mean():.3f}, size_norm.mean={size_norm.mean():.3f}")
        score = (1 - lam) * sym_norm + lam * size_norm

        return score
    
    def _calibrate_probabilities(self, probs: torch.Tensor, method: str = "temperature") -> torch.Tensor:
        """Calibrate overconfident probabilities to make them suitable for conformal prediction."""
        if not isinstance(probs, torch.Tensor):
            probs = torch.from_numpy(np.asarray(probs))
        probs = probs.float()
        
        # Check if probabilities are too extreme
        min_nonzero = probs[probs > 0].min() if (probs > 0).any() else 1e-9
        max_prob = probs.max()
        
        logger.info(f"[Calibrate] Input probs: min_nonzero={min_nonzero:.6f}, max={max_prob:.6f}")
        
        if method == "temperature":
            # Find temperature that spreads out the probabilities
            # Start with a lower temperature (higher value) to reduce confidence
            temp_candidates = [2.0, 3.0, 5.0, 10.0, 20.0]
            
            best_temp = 1.0
            best_score = float('inf')
            
            for temp in temp_candidates:
                # Convert back to logits (approximately) then apply temperature
                logits = torch.log(probs + 1e-9)
                temp_probs = F.softmax(logits / temp, dim=1)
                
                # Score based on how spread out the probabilities are
                entropy = -(temp_probs * torch.log(temp_probs + 1e-9)).sum(dim=1).mean()
                target_entropy = np.log(probs.shape[1]) * 0.3  # Target ~30% of max entropy
                score = abs(entropy - target_entropy)
                
                logger.info(f"[Calibrate] Temp={temp}: entropy={entropy:.3f}, target={target_entropy:.3f}, score={score:.3f}")
                
                if score < best_score:
                    best_score = score
                    best_temp = temp
            
            # Apply best temperature
            logits = torch.log(probs + 1e-9)
            calibrated_probs = F.softmax(logits / best_temp, dim=1)
            
            logger.info(f"[Calibrate] Selected temperature: {best_temp}")
            
        elif method == "label_smoothing":
            # Apply label smoothing to reduce overconfidence
            alpha = 0.1  # smoothing parameter
            num_classes = probs.shape[1]
            
            # Smooth the probabilities
            calibrated_probs = (1 - alpha) * probs + alpha / num_classes
            
            logger.info(f"[Calibrate] Applied label smoothing with alpha={alpha}")
        
        else:
            calibrated_probs = probs
        
        # Verify calibration worked
        cal_min = calibrated_probs[calibrated_probs > 0].min() if (calibrated_probs > 0).any() else 1e-9
        cal_max = calibrated_probs.max()
        logger.info(f"[Calibrate] Output probs: min_nonzero={cal_min:.6f}, max={cal_max:.6f}")
        
        return calibrated_probs

    @torch.no_grad()
    def _debug_conformal_calibration(self, target_size: int = 3) -> dict:
        """Debug the conformal calibration process step by step."""
        logger.info("[DEBUG] Starting conformal calibration debug...")
        
        idx_all = np.flatnonzero(self.labeled_pool)
        if idx_all.size < 20:
            logger.warning(f"[DEBUG] Too few labeled points: {idx_all.size}")
            return {"error": "too_few_labels"}
        
        # Use a subset for calibration
        m = min(idx_all.size, 200)
        cal_idx = np.random.choice(idx_all, size=m, replace=False)
        feats = self.features[cal_idx]
        
        # Get predictions from student model
        probs = self.model.get_probs(feats)
        if not isinstance(probs, torch.Tensor):
            probs = torch.from_numpy(np.asarray(probs))
        probs = probs.float()
        
        logger.info(f"[DEBUG] Raw probs shape: {probs.shape}")
        logger.info(f"[DEBUG] Raw probs range: [{probs.min():.6f}, {probs.max():.6f}]")
        logger.info(f"[DEBUG] Raw probs row sums (first 5): {probs.sum(dim=1)[:5]}")
        
        # Check if they look like probabilities
        row_sums = probs.sum(dim=1)
        looks_like_probs = (probs.min() >= -1e-6) and (probs.max() <= 1.1) and torch.all(torch.abs(row_sums - 1.0) <= 0.1)
        logger.info(f"[DEBUG] Looks like probabilities: {looks_like_probs}")
        
        # Apply _to_probs
        probs_normalized = self._to_probs(probs)
        logger.info(f"[DEBUG] After _to_probs range: [{probs_normalized.min():.6f}, {probs_normalized.max():.6f}]")
        logger.info(f"[DEBUG] After _to_probs row sums (first 5): {probs_normalized.sum(dim=1)[:5]}")
        
        # Compute conformal scores
        eps = 1e-9
        aI = -torch.log(probs_normalized.clamp_min(eps))
        logger.info(f"[DEBUG] Conformal scores (-log(p)) shape: {aI.shape}")
        logger.info(f"[DEBUG] Conformal scores range: [{aI.min():.6f}, {aI.max():.6f}]")
        logger.info(f"[DEBUG] Conformal scores mean: {aI.mean():.6f}")
        
        # Check for extreme values
        inf_mask = torch.isinf(aI)
        nan_mask = torch.isnan(aI)
        if inf_mask.any():
            logger.warning(f"[DEBUG] Found {inf_mask.sum()} infinite values in conformal scores!")
            logger.warning(f"[DEBUG] Corresponding probs: {probs_normalized[inf_mask]}")
        if nan_mask.any():
            logger.warning(f"[DEBUG] Found {nan_mask.sum()} NaN values in conformal scores!")
        
        # Analyze per-sample thresholds for target size
        C = aI.shape[1]
        sorted_scores_per_sample, _ = torch.sort(aI, dim=1)
        target_idx = min(target_size - 1, C - 1)
        empirical_thresholds = sorted_scores_per_sample[:, target_idx]
        
        logger.info(f"[DEBUG] Target size: {target_size}, target_idx: {target_idx}")
        logger.info(f"[DEBUG] Empirical thresholds range: [{empirical_thresholds.min():.6f}, {empirical_thresholds.max():.6f}]")
        logger.info(f"[DEBUG] Empirical thresholds mean: {empirical_thresholds.mean():.6f}")
        
        # Check different percentiles
        for percentile in [0.5, 0.8, 0.9, 0.95]:
            q_val = float(torch.quantile(empirical_thresholds, percentile).item())
            pred_sets = (aI <= q_val)
            avg_size = pred_sets.sum(dim=1).float().mean().item()
            logger.info(f"[DEBUG] At {percentile*100}th percentile (q={q_val:.4f}): avg set size = {avg_size:.2f}")
        
        # Test the current stored quantile if it exists
        if hasattr(self, 'q_image') and self.q_image is not None:
            current_sets = (aI <= self.q_image)
            current_avg_size = current_sets.sum(dim=1).float().mean().item()
            logger.info(f"[DEBUG] Current stored q_image={self.q_image:.4f}: avg set size = {current_avg_size:.2f}")
        
        # Check global quantile approach
        all_scores = aI.reshape(-1)
        target_fraction = target_size / C
        global_q = float(torch.quantile(all_scores, 1.0 - target_fraction).item())
        global_sets = (aI <= global_q)
        global_avg_size = global_sets.sum(dim=1).float().mean().item()
        logger.info(f"[DEBUG] Global approach (q={global_q:.4f}): avg set size = {global_avg_size:.2f}")
        
        return {
            "probs_shape": probs.shape,
            "probs_range": [float(probs.min()), float(probs.max())],
            "looks_like_probs": looks_like_probs,
            "scores_range": [float(aI.min()), float(aI.max())],
            "empirical_thresholds_range": [float(empirical_thresholds.min()), float(empirical_thresholds.max())],
            "has_inf": bool(inf_mask.any()),
            "has_nan": bool(nan_mask.any()),
            "current_q_image": getattr(self, 'q_image', None),
        }

    @torch.no_grad()
    def _fit_conformal_target_size_student(
        self,
        target_size: int = 3,
        subset_size: int = 1000,
        eps: float = 1e-9,
    ) -> dict:
        """
        Fit the conformal quantile q_image using the standard method.
        This replaces the flawed binary search.
        """
        idx_all = np.flatnonzero(self.labeled_pool)
        if idx_all.size < 20:
            logger.warning("[Conformal-Student] Too few labeled points for calibration")
            return {"ok": False, "reason": "too_few_labels"}

        m = int(min(idx_all.size, max(100, subset_size)))
        cal_idx = np.random.choice(idx_all, size=m, replace=False)
        feats = self.features[cal_idx]
        
        # Get and calibrate probabilities
        probs = self.model.get_probs(feats)
        probs = self._calibrate_probabilities(probs, method="temperature")
        probs = self._to_probs(probs)
        
        C = probs.shape[1]
        aI = -torch.log(probs.clamp_min(eps))

        # --- CORRECT CONFORMAL METHOD ---
        # 1. For each sample, find the score of the TRUE class.
        true_labels = torch.from_numpy(self.labels[cal_idx]).long()
        true_labels = true_labels.view(-1, 1) 
        conformal_scores = torch.gather(aI, 1, true_labels).squeeze()

        # 2. The quantile is the (1-alpha) percentile of these scores.
        #    alpha is the desired error rate (e.g., 0.1 for 90% coverage).
        alpha = 0.1
        q_level = np.ceil((m + 1) * (1 - alpha)) / m
        q_level = min(q_level, 1.0) # ensure it's <= 1
        
        self.q_image = float(torch.quantile(conformal_scores, q_level, interpolation='higher'))
        
        # --- VERIFICATION ---
        achieved_coverage = (conformal_scores <= self.q_image).float().mean().item()
        achieved_set_size = (aI <= self.q_image).sum(dim=1).float().mean().item()
        
        logger.info(
            f"[Conformal-Student] q_image={self.q_image:.4f} | "
            f"Coverage: {achieved_coverage:.2%} (target: {1-alpha:.0%}) | "
            f"Avg Set Size: {achieved_set_size:.2f}"
        )
        
        return {
            "ok": True,
            "q_image": self.q_image,
            "achieved_size": achieved_set_size,
            "achieved_coverage": achieved_coverage,
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

    # ---------- Main query method ----------
    def query(self, num_samples: int) -> NDArray[np.bool_]:
        """Select points to label based on multimodal uncertainty and diversity."""
        mask = np.zeros(len(self.features), dtype=bool)
        unlabeled_idx = np.flatnonzero(~self.labeled_pool)
        if num_samples <= 0 or len(unlabeled_idx) == 0:
            return mask

        # ---- subsequent rounds: CCMA ----
        pool_idxs = self._select_subpool()  # unlabeled-relative

        # --- Ablation study 1: Replace Diversity-based Subpool with a Large Random Pool) ---
        # The select_subpool() method is performing well than the random choice which increase to +1%
        # # Instead of _select_subpool(), we take a large, unbiased random sample.
        # # This ensures we don't prematurely discard informative outliers.
        # CANDIDATE_POOL_SIZE = 30000  # A large but computationally manageable number
        
        # num_candidates = min(len(unlabeled_idx), CANDIDATE_POOL_SIZE)
        
        # # These are indices relative to the unlabeled pool
        # candidate_idxs_unl = np.random.choice(len(unlabeled_idx), num_candidates, replace=False)

        # # The rest of the pipeline now operates on this better candidate pool
        # pool_idxs = candidate_idxs_unl

        if len(pool_idxs) == 0:
            return mask
        if pool_idxs.max(initial=-1) >= unlabeled_idx.size:
            raise RuntimeError(
                "[query] _select_subpool returned indices outside unlabeled range."
            )

        image_p, text_p = self._compute_modalities(pool_idxs)

        GI, GT = self._prediction_sets(
            image_p,
            text_p,
            qI_override=self.q_image, 
            qT_override=self.q_text,
            ensure_nonempty=True,
        )

        if self.debug:
            with torch.no_grad():
                sI = GI.sum(1).float().mean().item()
                sT = GT.sum(1).float().mean().item() if GT is not None else float("nan")
                logger.info(
                    f"[CCMA-debug] subpool mean |Γ_I|={sI:.2f}, |Γ_T|={sT:.2f} (C={GI.size(1)})"
                )
            try:
                self.ccma_set_stats(GI, GT)
            except Exception as e:
                logger.warning(f"[CCMA] stats logging skipped: {e}")

        score = self._conformal_score(GI, GT, lam=self.lam, image_probs=image_p)
       
        # Oversample top candidates
        topk = int(min(score.shape[0], max(num_samples * self.oversampling, num_samples)))
        top_idxs = torch.topk(score, topk).indices.cpu().numpy()  # indices into pool_idxs
        selected_pool = pool_idxs[top_idxs]  # unlabeled-relative

        # Final diversity selection in CLIP space if available
        feats_unl = (
            self.clip_image_embeds if self.clip_image_embeds is not None else self.features
        )[~self.labeled_pool]
        X = F.normalize(torch.from_numpy(feats_unl[selected_pool]).float(), dim=1).numpy()

        # --- Diversity pick via FAISS KMeans (assign+select) with sklearn fallback ---
        k = max(1, num_samples)
        picked_pool = None

        try:
            logger.info(f"[query] Using FAISS KMeans assign+select: {X.shape[0]} points, target {k} reps")

            # Make a float tensor on GPU if available (keeps CPU fallback cheap)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if isinstance(X, torch.Tensor):
                feats_tensor = X.detach().float().to(device)
            else:
                feats_tensor = torch.from_numpy(X).float().to(device)

            # (Optional but recommended for CLIP-like spaces)
            feats_tensor = F.normalize(feats_tensor, dim=1)

            # Use the robust function that returns indices into X and the actual cluster count
            sel_idx, k_used = cluster_and_select_gpu(
                feats_tensor,
                num_clusters=k,                 # request up to k clusters (may be reduced internally)
                num_samples_to_select=k,        # pick k representatives (trimmed/padded as needed)
            )
            sel_idx_np = sel_idx.cpu().numpy()

            # Map back to your selected_pool (sel_idx are indices into X / selected_pool)
            picked_pool = selected_pool[sel_idx_np]
            logger.info(f"[query] FAISS assign+select successful: selected {len(picked_pool)} reps (clusters used={k_used})")

        except Exception as e:
            logger.warning(f"[query] FAISS path failed: {e}. Falling back to sklearn KMeans.")

            try:
                # sklearn expects numpy
                X_np = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else X
                km = KMeans(n_clusters=k, n_init=10, max_iter=100, random_state=42)
                labels = km.fit_predict(X_np)

                # Choose the highest-scoring item per cluster (NOT the centroid)
                score_np = score[top_idxs].detach().cpu().numpy() if isinstance(score, torch.Tensor) else score[top_idxs]
                picked = []
                for c in range(k):
                    idx_c = np.where(labels == c)[0]
                    if idx_c.size == 0:
                        continue
                    best_local = idx_c[np.argmax(score_np[idx_c])]
                    picked.append(selected_pool[best_local])

                picked_pool = np.array(picked, dtype=np.int64)
                logger.info(f"[query] sklearn KMeans fallback successful: selected {len(picked_pool)} reps")

            except Exception as e2:
                logger.warning(f"[query] sklearn fallback also failed: {e2}. Falling back to top-k selection.")
                picked_pool = selected_pool[:num_samples]

        # Ensure we have valid results
        if picked_pool is None or len(picked_pool) == 0:
            logger.warning("[query] Both clustering methods failed, falling back to top-k selection")
            picked_pool = selected_pool[:num_samples]

        final_subpool = picked_pool[: min(len(picked_pool), num_samples)]
        mask[unlabeled_idx[final_subpool]] = True  # map unlabeled-relative -> absolute
        return mask
