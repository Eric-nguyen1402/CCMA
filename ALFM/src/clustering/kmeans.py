"""Implementation of K-Means++."""

from typing import Optional, Tuple

import faiss
import numpy as np
import torch
from numpy.typing import NDArray
from rich.progress import track
import logging

logger = logging.getLogger(__name__)

def faiss_pd(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x, y = x.numpy(), y.numpy()
    dist_matrix = faiss.pairwise_distances(x, y)
    return torch.from_numpy(dist_matrix)


def torch_pd(x: torch.Tensor, y: torch.Tensor, batch_size: int = 10240) -> torch.Tensor:
    x, y = x.cuda(), y.cuda()
    result = torch.zeros(x.shape[0], y.shape[0], device=x.device)

    for i in range(0, x.shape[0], batch_size):
        for j in range(0, y.shape[0], batch_size):
            x_batch = x[i : i + batch_size]
            y_batch = y[j : j + batch_size]

            dists = torch.cdist(x_batch.unsqueeze(0), y_batch.unsqueeze(0)).squeeze(0)
            result[i : i + x_batch.shape[0], j : j + y_batch.shape[0]] = dists

    return result


def kmeans_plus_plus_init(features: NDArray[np.float32], k: int) -> NDArray[np.int64]:
    centroids = []
    vectors = torch.from_numpy(features).cuda()
    n, d = vectors.shape

    # Choose the first centroid uniformly at random
    idx = np.random.randint(n)
    centroids.append(idx)

    # Compute the squared distance from all points to the centroid
    # pairwise_distances in FAISS returns the squared L2 distance
    centroid_vector = vectors[idx].view(1, -1)
    sq_dist = torch_pd(vectors, centroid_vector).ravel() ** 2
    sq_dist = sq_dist.clip(min=0).nan_to_num()  # avoid numerical errors

    # Choose the remaining centroids
    for _ in track(range(1, k), description="[green]K-Means++ init"):
        probabilities = sq_dist / torch.sum(sq_dist)
        idx = torch.multinomial(probabilities, 1).item()  # type: ignore[assignment]
        centroids.append(idx)

        # update the squared distances
        centroid_vector = vectors[idx].view(1, -1)
        new_dist = torch_pd(vectors, centroid_vector).ravel() ** 2
        new_dist = new_dist.clip(min=0).nan_to_num()  # avoid numerical errors

        # update the minimum squared distance
        sq_dist = torch.minimum(sq_dist, new_dist)

    return np.array(centroids)


def cluster_features(
    features: NDArray[np.float32],
    num_samples: int,
    weights: Optional[NDArray[np.float32]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_samples = int(num_samples)  # np int scalars cause problems with faiss

    # Ensure we don't request more clusters than points
    num_samples = min(num_samples, features.shape[0])

    kmeans = faiss.Kmeans(
        features.shape[1],
        num_samples,
        niter=100,
        gpu=1,
        verbose=True,
        min_points_per_centroid=1,
        max_points_per_centroid=512,
    )
    init_idx = kmeans_plus_plus_init(features, num_samples)
    kmeans.train(features, weights=weights, init_centroids=features[init_idx])

    sq_dist, cluster_idx = kmeans.index.search(features, 1)
    sq_dist = torch.from_numpy(sq_dist).ravel()
    cluster_idx = torch.from_numpy(cluster_idx).ravel()
    selected = torch.zeros(num_samples, dtype=torch.int64)

    # Track which clusters we've filled
    filled_clusters = 0
    
    for i in range(num_samples):
        idx = torch.nonzero(cluster_idx == i).ravel()
        
        # Check if this cluster has any points assigned
        if len(idx) > 0:
            min_idx = sq_dist[idx].argmin()  # point closest to the centroid
            selected[filled_clusters] = idx[min_idx]  # add that id to the selected set
            filled_clusters += 1
        # If cluster is empty, we skip it and don't increment filled_clusters

    # If some clusters were empty, trim the selected tensor
    if filled_clusters < num_samples:
        logger.warning(f"Only {filled_clusters} out of {num_samples} clusters had points assigned. Trimming selection.")
        selected = selected[:filled_clusters]
        
        # If we need more samples, randomly fill the remaining slots
        if filled_clusters < num_samples:
            remaining_points = torch.arange(features.shape[0])
            # Remove already selected points
            mask = torch.ones(features.shape[0], dtype=torch.bool)
            if filled_clusters > 0:
                mask[selected[:filled_clusters]] = False
            remaining_points = remaining_points[mask]
            
            # Randomly select additional points to reach target
            additional_needed = min(num_samples - filled_clusters, len(remaining_points))
            if additional_needed > 0:
                random_indices = torch.randperm(len(remaining_points))[:additional_needed]
                additional_selected = remaining_points[random_indices]
                selected = torch.cat([selected[:filled_clusters], additional_selected])

    return selected, cluster_idx

@torch.no_grad()
def cluster_and_select_gpu(
    features: torch.Tensor,
    num_clusters: int,
    num_samples_to_select: int,
) -> Tuple[torch.Tensor, int]:
    """
    K-Means with FAISS, then select multiple samples closest to each centroid.
    Crucially, avoids a separate (fragile) GPU flat-index search. Instead, it
    uses kmeans.index.search(features, 1) to get each point's assigned cluster
    and its distance to the cluster centroid, and picks the top-k per cluster.

    Args:
        features: (N, D) float tensor on any device; will be copied to CPU for FAISS.
        num_clusters: requested clusters (may be reduced internally for stability).
        num_samples_to_select: total number of samples to return.

    Returns:
        selected_indices (torch.long on the original device), actual_clusters (int)
    """
    assert features.dim() == 2, "features must be (N, D)"
    n, d = features.shape
    device = features.device

    # Determine a safe/actual cluster count (>=1, <=N)
    max_reasonable_clusters = max(1, min(num_clusters, n))
    # Keep at least ~10 pts per cluster if possible
    max_reasonable_clusters = min(max_reasonable_clusters, max(1, n // 10))
    # Hard cap to avoid GPU memory spikes
    max_reasonable_clusters = min(max_reasonable_clusters, 5000)
    k = max(1, max_reasonable_clusters)

    # Prepare data for FAISS (CPU float32, contiguous)
    x = np.ascontiguousarray(features.detach().float().cpu().numpy())

    # Train KMeans (prefer GPU if available in FAISS build)
    use_gpu = hasattr(faiss, "StandardGpuResources")
    kmeans = faiss.Kmeans(
        d,
        k,
        niter=20,                 # conservative, usually enough
        verbose=False,
        gpu=bool(use_gpu),
        min_points_per_centroid=1,
        max_points_per_centroid=max(512, (n // max(k, 1)) * 2),
    )
    kmeans.train(x)

    # Assign each point to nearest centroid & get squared distance to that centroid
    # NOTE: stays on the stable code path used during training
    sq_dist, cluster_idx = kmeans.index.search(x, 1)  # shapes (N,1), (N,1)
    sq_dist = sq_dist.ravel()
    cluster_idx = cluster_idx.ravel()

    # Decide how many to take per cluster (ceil to cover the request)
    samples_per_cluster = max(1, int(np.ceil(num_samples_to_select / k)))

    # For each cluster, take the closest 'samples_per_cluster' points (by sq_dist)
    picks = []
    for c in range(k):
        members = np.nonzero(cluster_idx == c)[0]
        if members.size == 0:
            continue
        order = np.argsort(sq_dist[members])
        take = members[order[:samples_per_cluster]]
        picks.append(take)

    if len(picks) == 0:
        # Extremely unlikely; fallback to random selection
        sel = torch.from_numpy(np.random.permutation(n)[:num_samples_to_select])
    else:
        sel = torch.from_numpy(np.concatenate(picks, axis=0)).unique()

    # Trim or pad to the requested count
    if sel.numel() >= num_samples_to_select:
        sel = sel[:num_samples_to_select]
    else:
        remaining = torch.tensor(
            np.setdiff1d(np.arange(n), sel.cpu().numpy()),
            dtype=torch.long,
        )
        need = num_samples_to_select - sel.numel()
        if remaining.numel() > 0 and need > 0:
            sel = torch.cat([sel, remaining[torch.randperm(remaining.numel())[:need]]], dim=0)

    return sel.to(device), int(k)