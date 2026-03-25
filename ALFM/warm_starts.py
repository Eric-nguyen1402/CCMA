# warm_starts.py
from __future__ import annotations
from typing import Optional, Tuple, Dict
import numpy as np
import torch
import torch.nn.functional as F
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _l2_normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.normalize(x, dim=dim)

@torch.no_grad()
def teacher_quality_gate(
    clip_image_embeds: Optional[np.ndarray],
    text_embeds: Optional[torch.Tensor],
    labels: Optional[np.ndarray],
    labeled_mask: Optional[np.ndarray],
    margin: float = 0.10,
) -> bool:
    """
    Returns True if zero-shot top-1 on the labeled slice exceeds chance + margin.
    """
    if clip_image_embeds is None or text_embeds is None or labels is None or labeled_mask is None:
        return False
    idx = np.flatnonzero(labeled_mask)
    if idx.size < 10:  # need a tiny seed to estimate
        return False

    X = torch.from_numpy(clip_image_embeds[idx]).float()
    X = _l2_normalize(X, dim=1)
    T = _l2_normalize(text_embeds.float(), dim=1)
    y = torch.from_numpy(labels[idx].reshape(-1)).long()

    logits = X @ T.T
    top1 = (logits.argmax(1) == y).float().mean().item()
    chance = 1.0 / T.size(0)
    return top1 >= chance + margin

@torch.no_grad()
def d2ds_warm_start(
    features_u: np.ndarray,              # (Nu, d), L2-normalized
    text_embeds: Optional[torch.Tensor], # (C, d), L2-normalized (or None)
    budget_B: int,
    alpha: float = 0.5,
    rounds_R: int = 5,
    candidate_factor: int = 8,
    rng: Optional[np.random.Generator] = None,
    batch_size: int = 16384,
) -> np.ndarray:
    """
    Dual-space D^2 Seeding (streaming k-means||-style) for warm start.
    Returns absolute unlabeled indices of size B.
    """
    assert features_u.ndim == 2
    Nu, d = features_u.shape
    B = int(budget_B)
    R = int(max(1, rounds_R))
    m_per_round = max(1, (candidate_factor * B) // R)
    rng = rng or np.random.default_rng()

    Z = torch.from_numpy(features_u).float()
    Z = torch.nan_to_num(Z)
    Z = _l2_normalize(Z, dim=1)

    # Teacher proximity (fixed per x)
    logger.info(f"[d2ds] Starting. Nu={Nu}, B={B}, alpha={alpha}, R={R}")
    if text_embeds is not None:
        T = _l2_normalize(text_embeds.float(), dim=1)  # (C, d)
        # compute min_c (1 - cos(z, t_c)) = 1 - max_c cos
        # via matrix multiply in chunks to save memory
        def maxcos_to_T(Zchunk: torch.Tensor) -> torch.Tensor:
            return (Zchunk @ T.T).amax(dim=1)
        maxcos_list = []
        for i in range(0, Nu, batch_size):
            maxcos_list.append(maxcos_to_T(Z[i : i + batch_size]))
        logger.info("[d2ds] Teacher embeds found. Calculating dT.")
        maxcos = torch.cat(maxcos_list, dim=0)
        dT = 1.0 - maxcos  # (Nu,)
        logger.info(f"[d2ds] dT calculated. Shape: {dT.shape}")
    else:
        dT = torch.zeros(Nu, dtype=torch.float32, device=Z.device)
        alpha = 1.0  # fall back to image-only

    # Initialize with one random seed
    mu = _l2_normalize(Z.mean(0, keepdim=True), dim=1)     # (1, d)
    cos_to_mu = (Z @ mu.T).squeeze(1)                      # (Nu,)
    first = int(torch.argmin(cos_to_mu).item())            # farthest from mean
    S = [first]
    logger.info(f"[d2ds] Initial seed: {S[0]}")
    # Track min distance to seeds in image space
    dS = torch.full((Nu,), float("inf"), dtype=torch.float32, device=Z.device)

    # helper to update dS with new seed indices (list of ints)
    def update_dS(new_ids):
        if not isinstance(new_ids, (list, np.ndarray)) or len(new_ids) == 0:
            logger.warning(f"[d2ds] update_dS called with invalid new_ids: {new_ids}")
            return
        logger.debug(f"[d2ds] update_dS with {len(new_ids)} new ids.")    
        try:
            # Ensure Zs is always 2D, even if new_ids has only one element
            Zs = Z[new_ids]
            if Zs.ndim == 1:
                Zs = Zs.unsqueeze(0)

            # compute 1 - cos(Z, Zs) = 1 - Z @ Zs.T
            # This can be done in a single matrix multiplication
            sim_matrix = Z @ Zs.T  # (Nu, |new_ids|)
            min_dist_to_new, _ = (1.0 - sim_matrix).min(dim=1)
            
            # Update the overall minimum distance
            dS.copy_(torch.minimum(dS, min_dist_to_new))
            logger.debug(f"[d2ds] update_dS successful. dS min/max: {dS.min():.3f}/{dS.max():.3f}")
        except Exception as e:
            logger.error(f"[d2ds] Exception in update_dS: {e}. Z.shape={Z.shape}, new_ids len={len(new_ids)}")
            # Fallback to safer loop for debugging
            for i in new_ids:
                center_vec = Z[i].unsqueeze(1) # (d, 1)
                dist = 1.0 - (Z @ center_vec).squeeze(1)
                dS.copy_(torch.minimum(dS, dist))

    update_dS([S[0]])

    # streaming rounds
    for r_idx in range(R):
        logger.info(f"[d2ds] Round {r_idx+1}/{R}")
        Delta = alpha * dS + (1.0 - alpha) * dT  # (Nu,)
        # probability ∝ Δ^2; avoid zeros
        probs = (Delta.clamp_min(1e-12) ** 2).cpu().numpy()
        probs_sum = probs.sum()
        if not np.isfinite(probs_sum) or probs_sum <= 0:
            logger.warning("[d2ds] Invalid probabilities in sampling. Falling back to uniform.")
            probs = np.ones_like(probs) / len(probs)
        else:
            probs /= probs_sum
        cand_size = min(m_per_round, Nu - len(S))
        if cand_size <= 0:
            break
        cand = rng.choice(Nu, size=cand_size, replace=False, p=probs)
        S.extend(cand.tolist())
        update_dS(cand.tolist())

    # Reduce candidates to B via medoids (closest to k-means centers)
    # We cluster only the |S|-sized subset; a simple k-means on these is small.
    logger.info(f"[d2ds] Candidate reduction. |S|={len(S)}")
    Suniq = np.unique(np.asarray(S, dtype=np.int64))
    Zc = Z[Suniq]  # (M, d)
    logger.info(f"[d2ds] Unique candidates: {Zc.shape[0]}. Starting k-means++.")
    with torch.no_grad():
        # sim_to_cand: (Nu, M)
        sim_to_cand = Z @ Zc.T
        assign_cand = sim_to_cand.argmax(dim=1)                    # (Nu,)
        W = torch.bincount(assign_cand, minlength=Zc.size(0)).float()  # (M,)
        # Avoid zero weights (rare but possible)
        W = torch.clamp(W, min=1.0)
    # lightweight k-means on Zc -> B clusters
    # We'll do Lloyd for small M, then return medoids (indices in original space)
    B_eff = min(B, Zc.size(0))
    # k-means++ init on Zc
    cent_ids = [int(rng.integers(0, Zc.size(0)))]
    cvecs = [Zc[cent_ids[0]].clone()]
    # --- LIKELY FIX ---
    # The mat-vec product Zc @ cvecs[0] results in a 1D tensor. Calling .squeeze(1) is invalid.
    # It should be removed as the result is already the correct shape.
    logger.debug(f"[d2ds] k-means++ init: Zc shape={Zc.shape}, cvecs[0] shape={cvecs[0].shape}")
    dmin = 1.0 - (Zc @ cvecs[0]) # No squeeze needed
    logger.debug(f"[d2ds] k-means++ init: dmin shape={dmin.shape}")

    for i in range(1, B_eff):
        p = (W * dmin.clamp_min(1e-12).pow(2)).cpu().numpy()
        p_sum = p.sum()
        if not np.isfinite(p_sum) or p_sum <= 0:
            logger.warning(f"[d2ds] k-means++ invalid probabilities at step {i}. Using uniform.")
            p = np.ones_like(p) / p.size
        else:
            p /= p_sum
        nxt = int(rng.choice(Zc.size(0), p=p))
        cent_ids.append(nxt)
        cvecs.append(Zc[nxt].clone())
        
        logger.debug(f"[d2ds] k-means++ step {i}: Zc shape={Zc.shape}, cvecs[-1] shape={cvecs[-1].shape}")
        d_new = 1.0 - (Zc @ cvecs[-1]) # No squeeze needed
        dmin = torch.minimum(dmin, d_new)
        logger.debug(f"[d2ds] k-means++ step {i}: dmin shape={dmin.shape}")

    C = torch.stack(cvecs, dim=0)  # (B_eff, d)
    logger.info(f"[d2ds] k-means++ done. Starting Lloyd steps. C shape={C.shape}")
    for i in range(10):  # a few Lloyd steps (small M)
        # assign
        sim = Zc @ C.T
        assign = sim.argmax(dim=1)  # (M,)
        # update
        newC = []
        for k in range(B_eff):
            idx = (assign == k).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                newC.append(C[k])
            else:
                wk = W[idx].unsqueeze(1)
                mean_k = (Zc[idx] * wk).sum(dim=0, keepdim=True) / (wk.sum() + 1e-12)
                newC.append(_l2_normalize(mean_k, dim=1).squeeze(0))
        newC = torch.stack(newC, dim=0)
        logger.debug(f"[d2ds] Lloyd step {i}: C updated.")
        if torch.allclose(newC, C, atol=1e-4):
            logger.info(f"[d2ds] Lloyd converged after {i+1} steps.")
            break
        C = newC

    # pick medoid per cluster
    logger.info("[d2ds] Picking medoids from clusters.")
    sim = Zc @ C.T
    assign = sim.argmax(dim=1)
    picked = []
    picked_set = set()
    for k in range(B_eff):
        idx = (assign == k).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        # medoid = argmax cosine to center
        score = ( (Zc[idx] @ C[k]) * W[idx] )
        best_local = idx[score.argmax()]
        cand_abs = int(Suniq[int(best_local)])
        if cand_abs not in picked_set:
            picked.append(cand_abs)
            picked_set.add(cand_abs)
    if len(picked) < B_eff:
        logger.info(f"[d2ds] Backfilling {B - len(picked)} points.")
        # backfill with highest-Delta points not yet picked
        Delta = alpha * dS + (1.0 - alpha) * dT
        order = torch.argsort(-Delta).cpu().numpy().tolist()
        for j in order:
            if j not in picked:
                picked.append(j)
            if len(picked) >= B:
                break
    logger.info(f"[d2ds] Final picks: {len(picked)}")
    return np.asarray(picked[:B], dtype=np.int64)

@torch.no_grad()
def tcfl_warm_start(
    features_u: np.ndarray,               # (Nu, d), L2
    text_embeds: Optional[torch.Tensor],  # (C, d), L2 (or None)
    budget_B: int,
    beta: float = 0.5,
    lazy_factor: float = 0.5,             # controls how aggressively to recompute bounds
) -> np.ndarray:
    """
    Submodular Teacher-Calibrated Facility Location via lazy greedy.
    Returns absolute unlabeled indices of size B.
    """
    Z = torch.from_numpy(features_u).float()
    Z = _l2_normalize(Z, dim=1)
    Nu, d = Z.shape
    B = min(budget_B, Nu)

    if text_embeds is not None:
        T = _l2_normalize(text_embeds.float(), dim=1)  # (C, d)
        # y*(x) and alignment matrix A = T[y*(x)]
        logits = Z @ T.T
        y_star = logits.argmax(dim=1)                  # (Nu,)
        A = T[y_star]                                  # (Nu, d)
    else:
        beta = 0.0
        A = torch.zeros_like(Z)

    S = []
    # current best gain per u: g(u) = max_{s∈S} sim(u,s) + beta * align(u,s)
    best = torch.full((Nu,), -float("inf"))
    # priority queue (lazy): store tuples (-bound, idx). We'll recompute if popped.
    import heapq
    pq = []
    # initial bounds: recompute exact ΔF for a small random subset to set a bar
    init_idx = np.random.choice(Nu, size=min(256, Nu), replace=False)
    base_gain = 0.0
    for j in init_idx:
        # ΔF(j|∅) = sum_u max(0, sim(u,j)+βalign(u,j))
        sim = (Z @ Z[j]) + beta * (A @ Z[j])
        gain = torch.clamp(sim, min=0.0).sum().item()
        base_gain = max(base_gain, gain)
        heapq.heappush(pq, (-gain, int(j)))

    # Push the rest with a looser initial bound
    loose_bound = -lazy_factor * base_gain
    for j in range(Nu):
        if j in init_idx:
            continue
        heapq.heappush(pq, (loose_bound, int(j)))  # negative when popped

    # Greedy selection
    for _ in range(B):
        while True:
            neg_bound, j = heapq.heappop(pq)
            # exact recompute
            sim = (Z @ Z[j]) + beta * (A @ Z[j])   # (Nu,)
            # marginal improvement over current best
            marg = torch.clamp(sim - best, min=0.0).sum().item()
            # check if bound was tight enough
            if -neg_bound >= marg - 1e-6:
                # accept
                break
            else:
                # push back with tighter bound
                heapq.heappush(pq, (-marg, j))
        # commit j
        S.append(j)
        # update best
        best = torch.maximum(best, sim)

    return np.asarray(S, dtype=np.int64)

@torch.no_grad()
def kmeanspp_warm_start_clip(
    clip_like: np.ndarray,          # (N, D) float32
    labeled_pool: np.ndarray,       # (N,) bool
    budget_B: int,
    lloyd_steps: int = 0,
    seed: int = 0,
) -> np.ndarray:
    """
    Cosine k-means++ seeding in CLIP space.
    Returns ABSOLUTE indices into the dataset (shape: (B,)).
    """
    assert clip_like.ndim == 2, f"clip_like must be (N,D), got {clip_like.shape}"
    assert labeled_pool.ndim == 1 and labeled_pool.dtype == bool, "labeled_pool must be boolean mask"
    N, D = clip_like.shape
    unl = np.flatnonzero(~labeled_pool)
    if budget_B <= 0 or unl.size == 0:
        return np.asarray([], dtype=np.int64)

    # Work in unlabeled CLIP space
    Z = torch.from_numpy(clip_like[unl]).float()
    Z = torch.nan_to_num(Z)
    Z = F.normalize(Z, dim=1)                 # (Nu, D)
    Nu = Z.size(0)
    K = min(budget_B, Nu)

    rng = np.random.default_rng(seed)

    # ----- pick first center: farthest from mean (more stable than random) -----
    mu = F.normalize(Z.mean(0, keepdim=True), dim=1)  # (1, D)
    cos_to_mu = (Z @ mu.T).squeeze(-1)                # (Nu,)
    first = int(torch.argmin(cos_to_mu).item())

    centers_rel = [first]

    # initialize min "distance" (cosine) to first center
    sim = Z @ Z[first]                    # (Nu,)
    min_d = (1.0 - sim).clamp_min(0.0)    # (Nu,) in [0, 2]

    # ----- k-means++ seeding -----
    # Avoid duplicate centers in degenerate cases
    chosen = np.zeros(Nu, dtype=bool)
    chosen[first] = True
    for _ in range(1, K):
        probs = (min_d + 1e-12).cpu().numpy()
        s = probs.sum()
        if s <= 0:
            remaining = np.flatnonzero(~chosen)
            if remaining.size == 0:
                break
            j = int(rng.choice(remaining))
        else:
            probs /= s
            j = int(rng.choice(Nu, p=probs))
            if chosen[j]:
                # very unlikely if probs>0, but guard anyway
                remaining = np.flatnonzero(~chosen)
                if remaining.size == 0:
                    break
                j = int(rng.choice(remaining))
        chosen[j] = True
        centers_rel.append(j)
        sim_j = Z @ Z[j]
        d_j = (1.0 - sim_j).clamp_min(0.0)
        min_d = torch.minimum(min_d, d_j)

    picks_abs = unl[np.array(centers_rel, dtype=np.int64)]

    # ----- optional Lloyd refinement with cosine means (keeps indices via medoids) -----
    if lloyd_steps > 0 and K > 0:
        C = F.normalize(Z[centers_rel].clone(), dim=1)   # (K, D)
        for _ in range(lloyd_steps):
            sims = Z @ C.T                                # (Nu, K)
            assign = torch.argmax(sims, dim=1)           # (Nu,)
            # update centers as normalized means
            for k in range(K):
                idxs = (assign == k).nonzero(as_tuple=False).squeeze(-1)
                if idxs.numel() > 0:
                    C[k:k+1] = F.normalize(Z[idxs].mean(0, keepdim=True), dim=1)
        # convert means back to medoid indices (so we return ABS indices)
        sims = Z @ C.T                                    # (Nu, K)
        centers_rel = torch.argmax(sims, dim=0).cpu().numpy().astype(np.int64)  # (K,)
        centers_rel = np.unique(centers_rel)
        if centers_rel.size < K:
            need = K - centers_rel.size
            remaining = np.setdiff1d(np.arange(Nu), centers_rel, assume_unique=False)
            if remaining.size > 0:
                fill = rng.choice(remaining, size=min(need, remaining.size), replace=False)
                centers_rel = np.concatenate([centers_rel, fill])
        picks_abs = unl[centers_rel]

    return picks_abs