import numpy as np
from collections import Counter, defaultdict
from sklearn.neighbors import NearestNeighbors
from typing import Sequence, Iterable


def _ensure_numpy(emb):
    if not isinstance(emb, np.ndarray):
        emb = np.asarray(emb)
    return emb


def precision_at_k_from_labels(query_label, neighbor_labels: Sequence, k: int) -> float:
    neighbor_labels_k = neighbor_labels[:k]
    return sum(1 for l in neighbor_labels_k if l == query_label) / float(k)


def average_precision_for_query(query_label, neighbor_labels: Iterable) -> float:
    """
    neighbor_labels: iterable of labels sorted by predicted relevance (closest first)
    returns average precision (binary relevance: label==query_label)
    """
    hits = 0
    sum_precisions = 0.0
    for i, lbl in enumerate(neighbor_labels, start=1):
        if lbl == query_label:
            hits += 1
            sum_precisions += hits / i
    if hits == 0:
        return 0.0
    return sum_precisions / hits


def compute_label_counts(labels: Sequence) -> Counter:
    return Counter(labels)


def evaluate_per_dataset_knn(embeddings,
                        labels: Sequence,
                        k_list=(1, 5, 10),
                        metric='cosine',
                        sample_size: int | None = None,
                        random_state: int = 0,
                        dataset_ids: Sequence | None = None):
    """
    Compute k-NN retrieval metrics (global + optional per-dataset).

    Returns dict with:
    {
        "global": {k -> metrics dict},
        "per_dataset": {dataset_id -> {k -> metrics dict}}
    }

    - embeddings: (N, D) array-like
    - labels: sequence length N (can be ints, strings, etc.)
    - k_list: iterable of k values
    - metric: 'cosine' or 'euclidean'
    - sample_size: optionally evaluate on a random subset (without replacement)
    - dataset_ids: sequence length N, optional grouping key for dataset-wise metrics
    """
    embeddings = _ensure_numpy(embeddings)
    N = embeddings.shape[0]
    if N == 0:
        raise ValueError("Empty embeddings array")

    labels = np.asarray(labels, dtype=object)
    if dataset_ids is not None:
        dataset_ids = np.asarray(dataset_ids, dtype=object)
        if len(dataset_ids) != N:
            raise ValueError("dataset_ids must have same length as embeddings")

    rng = np.random.default_rng(random_state)

    # optional sampling
    if sample_size is not None and sample_size < N:
        idx = rng.choice(N, size=sample_size, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]
        if dataset_ids is not None:
            dataset_ids = dataset_ids[idx]
        N = embeddings.shape[0]

    # sanitize and sort k_list
    k_list = sorted(int(k) for k in set(k_list))
    max_k = max(k_list)
    max_k = min(max_k, max(0, N - 1))
    if max_k == 0:
        results = {k: {"topk_acc": 0.0, "precision@k": 0.0,
                       "recall@k": 0.0, "mAP": 0.0}
                   for k in k_list}
        return {"global": results, "per_dataset": {}}

    # request one extra neighbor to allow removing self
    n_neighbors_request = min(N, max_k + 1)

    nn = NearestNeighbors(n_neighbors=n_neighbors_request,
                          metric=metric,
                          n_jobs=-1)
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings, return_distance=True)

    # Build neighbor indices excluding self
    sentinel = -1
    neighbor_indices_no_self = np.full((N, max_k), sentinel, dtype=int)
    for i in range(N):
        row = indices[i].tolist()
        filtered = [x for x in row if x != i]
        take = filtered[:max_k]
        neighbor_indices_no_self[i, :len(take)] = take

    # Build neighbor labels matrix
    neighbor_labels_mat = np.empty((N, max_k), dtype=object)
    neighbor_labels_mat[:, :] = None
    for i in range(N):
        for j in range(max_k):
            idx = neighbor_indices_no_self[i, j]
            if idx != sentinel:
                neighbor_labels_mat[i, j] = labels[idx]

    # Precompute label counts for recall denominator
    label_counts = compute_label_counts(labels)

    # Precompute per-query AP
    aps = np.array([
        average_precision_for_query(labels[i], neighbor_labels_mat[i, :].tolist())
        for i in range(N)
    ], dtype=float)

    def compute_metrics_for_indices(query_indices):
        # mAP for just these queries
        subset_aps = aps[list(query_indices)]
        subset_mAP = float(np.mean(subset_aps)) if len(subset_aps) > 0 else 0.0

        subset_results = {}
        for k in k_list:
            if k <= 0:
                subset_results[k] = {
                    "topk_acc": 0.0,
                    "precision@k": 0.0,
                    "recall@k": 0.0,
                    "mAP": subset_mAP,   # <-- use subset mAP
                }
                continue

            kk = min(k, max_k)

            # top-k accuracy
            topk_hits = np.array(
                [any(neighbor_labels_mat[i, :kk] == labels[i]) for i in query_indices],
                dtype=float
            )
            topk_acc = float(topk_hits.mean())

            # precision@k
            precs = []
            for i in query_indices:
                topk_labels = neighbor_labels_mat[i, :kk]
                num_correct = sum(1 for x in topk_labels if x == labels[i])
                precs.append(num_correct / float(k))
            precision_at_k = float(np.mean(precs))

            # recall@k
            recalls = []
            for i in query_indices:
                total_relevant = label_counts[labels[i]] - 1
                if total_relevant <= 0:
                    continue
                topk_labels = neighbor_labels_mat[i, :kk]
                num_correct = sum(1 for x in topk_labels if x == labels[i])
                recalls.append(num_correct / float(total_relevant))
            recall_at_k = float(np.mean(recalls)) if recalls else 0.0

            subset_results[k] = {
                "topk_acc": topk_acc,
                "precision@k": precision_at_k,
                "recall@k": recall_at_k,
                "mAP": subset_mAP,    # <-- use subset mAP
            }
        return subset_results

    # ---- Global metrics ----
    global_results = compute_metrics_for_indices(range(N))

    # ---- Per-dataset metrics ----
    per_dataset_results = {}
    if dataset_ids is not None:
        dataset_to_indices = defaultdict(list)
        for i, ds in enumerate(dataset_ids):
            dataset_to_indices[ds].append(i)
        for ds, idxs in dataset_to_indices.items():
            per_dataset_results[ds] = compute_metrics_for_indices(idxs)

    return {
        "global": global_results,
        "per_dataset": per_dataset_results
    }