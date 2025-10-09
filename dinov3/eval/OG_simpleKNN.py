import numpy as np
from collections import Counter
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

def _ensure_numpy(emb):
    if not isinstance(emb, np.ndarray):
        emb = np.asarray(emb)
    return emb

def precision_at_k_from_labels(query_label, neighbor_labels, k):
    neighbor_labels_k = neighbor_labels[:k]
    return sum(1 for l in neighbor_labels_k if l == query_label) / k

def average_precision_for_query(query_label, neighbor_labels):
    """
    neighbor_labels: iterable of labels sorted by predicted relevance (descending)
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

def compute_label_counts(labels):
    cnt = Counter(labels)
    return cnt  # mapping label -> count

def evaluate_simple_knn(embeddings, labels, k_list=[1,5,10], metric='cosine', sample_size=None, random_state=0):
    """
    embeddings: np.ndarray (N, D)
    labels: list/array length N
    k_list: list of k values to compute metrics for
    metric: 'cosine' (recommended) or 'euclidean'
    sample_size: if not None -> randomly sample that many points for evaluation (both index and queries)
    """
    print("Using old eval")
    embeddings = _ensure_numpy(embeddings)
    N = embeddings.shape[0]
    labels = np.array(labels)
    rng = np.random.default_rng(random_state)

    # optionally sample a subset (sample without replacement)
    if sample_size is not None and sample_size < N:
        idx = rng.choice(N, size=sample_size, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]
        N = embeddings.shape[0]

    # For cosine use normalized vectors with Euclidean NN:  cosine(a,b) = 1 - (a·b) if normalized, but sklearn supports metric='cosine'
    if metric == 'cosine':
        # sklearn's NearestNeighbors with metric='cosine' returns smaller distances for more similar vectors (cosine distance)
        nn = NearestNeighbors(n_neighbors=max(k_list)+1, metric='cosine', n_jobs=-1)
    else:
        nn = NearestNeighbors(n_neighbors=max(k_list)+1, metric='euclidean', n_jobs=-1)

    nn.fit(embeddings)
    
    # kneighbors returns distances, indices
    # we query all points (each point will return itself as the nearest -> exclude)
    distances, indices = nn.kneighbors(embeddings, return_distance=True)
    # remove self (first neighbor is usually itself at index 0)
    indices = indices[:, 1:]  # shape (N, max_k)
    
    results = {}
    max_k = max(k_list)
    # precompute neighbor labels for each query
    neighbor_labels_mat = labels[indices]  # shape (N, max_k)
    
    # compute counts per label for recall normalization
    label_counts = compute_label_counts(labels)
    
    for k in tqdm(k_list):
        topk = neighbor_labels_mat[:, :k]
        # top-1 accuracy (if k==1)
        top1_acc = np.mean([1 if topk[i,0] == labels[i] else 0 for i in range(N)])
        # precision@k
        precs = [precision_at_k_from_labels(labels[i], topk[i], k) for i in range(N)]
        precision_at_k = float(np.mean(precs))
        # recall@k: (#relevant retrieved)/ (#relevant in dataset - 1) [exclude the query itself]
        recalls = []
        for i in range(N):
            total_relevant = label_counts[labels[i]] - 1  # exclude query
            if total_relevant <= 0:
                recalls.append(0.0)
            else:
                recalls.append(sum(1 for l in topk[i] if l == labels[i]) / total_relevant)
        recall_at_k = float(np.mean(recalls))
        # mAP (average precision across queries)
        aps = [average_precision_for_query(labels[i], list(neighbor_labels_mat[i])) for i in range(N)]
        mAP = float(np.mean(aps))
        results[k] = {"top1_acc": top1_acc, "precision@k": precision_at_k, "recall@k": recall_at_k, "mAP": mAP}
    return results
