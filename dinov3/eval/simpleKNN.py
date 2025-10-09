import numpy as np
from collections import Counter
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


def evaluate_simple_knn(embeddings,
                        labels: Sequence,
                        k_list=(1, 5, 10),
                        metric='cosine',
                        sample_size: int | None = None,
                        random_state: int = 0):
    """
    Compute k-NN retrieval metrics.

    Returns dict mapping k -> {"topk_acc", "precision@k", "recall@k", "mAP"}.
    mAP is computed using the ranked neighbor list truncated at max(k_list).
    - embeddings: (N, D) array-like
    - labels: sequence length N (can be ints, strings, etc.)
    - k_list: iterable of k values
    - metric: 'cosine' or 'euclidean'
    - sample_size: optionally evaluate on a random subset (without replacement)
    """
    embeddings = _ensure_numpy(embeddings)
    N = embeddings.shape[0]
    if N == 0:
        raise ValueError("Empty embeddings array")

    # make labels an object array so sentinel values are easy to use
    labels = np.asarray(labels, dtype=object)
    rng = np.random.default_rng(random_state)

    # optional sampling
    if sample_size is not None and sample_size < N:
        idx = rng.choice(N, size=sample_size, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]
        N = embeddings.shape[0]

    # sanitize and sort k_list
    k_list = sorted(int(k) for k in set(k_list))
    max_k = max(k_list)
    # can't ask for more neighbors than N-1 (excluding self)
    max_k = min(max_k, max(0, N - 1))
    if max_k == 0:
        # trivial case: no neighbors to inspect
        results = {}
        for k in k_list:
            results[k] = {"topk_acc": 0.0, "precision@k": 0.0, "recall@k": 0.0, "mAP": 0.0}
        return results

    # request one extra neighbor to allow removing self (we will request up to N neighbors including self)
    n_neighbors_request = min(N, max_k + 1)

    if metric == 'cosine':
        nn = NearestNeighbors(n_neighbors=n_neighbors_request, metric='cosine', n_jobs=-1)
    else:
        nn = NearestNeighbors(n_neighbors=n_neighbors_request, metric='euclidean', n_jobs=-1)

    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings, return_distance=True)  # shape (N, n_neighbors_request)

    # Build neighbor indices excluding self per-row. Pad with -1 sentinel if fewer than max_k neighbors remain.
    sentinel = -1
    neighbor_indices_no_self = np.full((N, max_k), sentinel, dtype=int)

    for i in range(N):
        row = indices[i].tolist()
        # remove occurrences of self index (there may be duplicates or self not at position 0 in degenerate cases)
        filtered = [x for x in row if x != i]
        # take up to max_k neighbors
        take = filtered[:max_k]
        neighbor_indices_no_self[i, :len(take)] = take

    # Build neighbor labels matrix (object dtype), missing neighbors = None
    neighbor_labels_mat = np.empty((N, max_k), dtype=object)
    neighbor_labels_mat[:, :] = None
    for i in range(N):
        for j in range(max_k):
            idx = neighbor_indices_no_self[i, j]
            if idx != sentinel:
                neighbor_labels_mat[i, j] = labels[idx]
            else:
                neighbor_labels_mat[i, j] = None

    # Precompute label counts for recall denominator (exclude query itself)
    label_counts = compute_label_counts(labels)

    # Precompute per-query AP using the truncated neighbor list (length max_k)
    aps = [average_precision_for_query(labels[i], neighbor_labels_mat[i, :].tolist()) for i in range(N)]
    global_mAP = float(np.mean(aps))

    results = {}
    for k in k_list:
        if k <= 0:
            results[k] = {"topk_acc": 0.0, "precision@k": 0.0, "recall@k": 0.0, "mAP": global_mAP}
            continue
        kk = min(k, max_k)
        # top-k accuracy: fraction of queries where any of top-k neighbors has the same label
        topk_hits = np.array([any(neighbor_labels_mat[i, :kk] == labels[i]) for i in range(N)], dtype=float)
        topk_acc = float(topk_hits.mean())

        # precision@k: average over queries of (# correct in top-k)/k
        precs = []
        for i in range(N):
            topk_labels = neighbor_labels_mat[i, :kk]
            num_correct = sum(1 for x in topk_labels if x == labels[i])
            # note: we divide by requested k (not by available neighbors), staying consistent with precision@k definition
            precs.append(num_correct / float(k))
        precision_at_k = float(np.mean(precs))

        # recall@k: average over queries that have at least one other item of same label (exclude queries
        # where label_counts[label] == 1)
        recalls = []
        valid_mask = []
        for i in range(N):
            total_relevant = label_counts[labels[i]] - 1  # exclude the query itself
            if total_relevant <= 0:
                # skip this query from recall averaging
                valid_mask.append(False)
                continue
            topk_labels = neighbor_labels_mat[i, :kk]
            num_correct = sum(1 for x in topk_labels if x == labels[i])
            recalls.append(num_correct / float(total_relevant))
            valid_mask.append(True)
        if len(recalls) == 0:
            recall_at_k = 0.0
        else:
            recall_at_k = float(np.mean(recalls))

        results[k] = {
            "topk_acc": topk_acc,
            "precision@k": precision_at_k,
            "recall@k": recall_at_k,
            "mAP": global_mAP,
        }

    return results
