import numpy as np
import pandas as pd  # only used inside helper; fine to keep
from functools import lru_cache


def build_triplet_lists_with_paths(X, clusterer):
    """
    Positives: parent centroids along each point's path (depth > 0).
    Negatives (for point i):
      - Outside i's top (depth-0) cluster: include TOP centroids of those clusters.
      - Inside the same top cluster:
          * For depths 0..D* (D* = deepest non-singleton depth for i):
              - include TRIMMED-LEAF nodes (no children with >=2 points):
                  · singleton leaf  -> raw point vector
                  · multi-point leaf -> node centroid
          * Additionally at depth == D*:
              - include NON-LEAF node centroids (has at least one child with >=2 points)
    Paths: trimmed ancestor chains (singleton-child edges removed). The final step includes
           'negatives_used' for QA (with depth & type).
    """
    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0]
    df = clusterer.condensed_tree_.to_pandas()[['parent','child','lambda_val','child_size']].copy()
    df['parent'] = df['parent'].astype(int)
    df['child']  = df['child'].astype(int)

    # --- adjacency from condensed tree ---
    children, child_to_parent, child_lambda = {}, {}, {}
    for p, c, lam, _ in df[['parent','child','lambda_val','child_size']].itertuples(index=False):
        children.setdefault(p, []).append(c)
        child_to_parent[c] = p
        child_lambda[c]    = float(lam)

    @lru_cache(None)
    def leaves_under(node: int) -> tuple:
        if node < n:  # original data point
            return (node,)
        out = []
        for ch in children.get(node, []):
            out.extend(leaves_under(ch))
        return tuple(sorted(out))

    @lru_cache(None)
    def centroid(node: int) -> np.ndarray:
        idx = np.fromiter(leaves_under(node), dtype=int)
        return X[idx].mean(axis=0)

    # ---- TRIMMED children: ignore children with size==1 (singleton edges are excluded) ----
    @lru_cache(None)
    def trimmed_children(node: int) -> tuple:
        chs = []
        for ch in children.get(node, []):
            if len(leaves_under(ch)) >= 2:  # keep only non-singleton children
                chs.append(int(ch))
        return tuple(sorted(chs))

    def is_trimmed_leaf(node: int) -> bool:
        return len(trimmed_children(node)) == 0

    # --- build trimmed paths (skip edges where child has size 1) ---
    paths, used_nodes = {}, set()
    for i in range(n):
        chain, node = [], i
        while node in child_to_parent:
            parent = child_to_parent[node]
            child_sz  = len(leaves_under(node))
            parent_sz = len(leaves_under(parent))
            if child_sz == 1:                 # exclude singleton child edges
                node = parent
                continue
            chain.append({
                'parent_id':       int(parent),
                'child_id':        int(node),
                'lambda_leave':    float(child_lambda[node]),
                'parent_size':     parent_sz,
                'child_size':      child_sz,
                'parent_centroid': centroid(parent),   # will be None at depth 0
                'child_centroid':  centroid(node),
            })
            used_nodes.update((int(parent), int(node)))
            node = parent
        chain.reverse()
        for d, step in enumerate(chain):
            step['depth'] = d
            if d == 0:
                step['parent_centroid'] = None
        paths[i] = chain

    # --- nodes present at each depth (from trimmed paths) ---
    depth_nodes = {}
    for i in range(n):
        for d, step in enumerate(paths[i]):
            depth_nodes.setdefault(d, set()).add(int(step['child_id']))

    # top (depth-0) nodes and membership
    top_nodes  = set(map(int, depth_nodes.get(0, set())))
    top_leaves = {t: set(map(int, leaves_under(t))) for t in top_nodes}

    # node -> its top ancestor (by leaf-set inclusion)
    nodes_we_might_use = set(used_nodes) | set().union(*depth_nodes.values()) | top_nodes
    node_to_top = {}
    for node in nodes_we_might_use:
        s = set(map(int, leaves_under(node)))
        mapped = next((t for t, Tset in top_leaves.items() if s.issubset(Tset)), None)
        node_to_top[node] = node if mapped is None else mapped

    # handy dicts
    node_centroid = {node: centroid(node) for node in nodes_we_might_use}
    node_leaves   = {node: np.array(leaves_under(node), dtype=int) for node in nodes_we_might_use}

    # --- outputs ---
    positives = [[] for _ in range(n)]
    lambdas   = [[] for _ in range(n)]
    negatives = [[] for _ in range(n)]
    neg_lambdas = [[] for _ in range(n)]

    _ = max(depth_nodes.keys()) if depth_nodes else -1 # max_depth

    for i in range(n):
        steps = paths[i]
        if not steps:
            continue

        # positives & lambdas (skip depth 0)
        for d, s in enumerate(steps):
            if d == 0:
                continue
            positives[i].append(s['parent_centroid'])
            lambdas[i].append(float(s['lambda_leave']))

        # deepest non-singleton depth for this point
        D_star = len(steps) - 1
        my_top_node  = int(steps[0]['child_id'])
        my_child_at_depth = {s['depth']: int(s['child_id']) for s in steps}

        neg_node_ids, neg_point_idx = set(), set()
        negatives_used, seen = [], set()

        # A) Outside my top cluster: add each other TOP centroid once
        for t in top_nodes:
            if t == my_top_node:
                continue
            sig = ('top', int(t))
            if sig not in seen:
                seen.add(sig)
                neg_node_ids.add(int(t))
                negatives_used.append({'depth': 0, 'type': 'top_centroid',
                                       'top_node_id': int(t),
                                       'vector': node_centroid[int(t)]})

        # B) Inside same top cluster: depths 0..D* (≤ D*): include TRIMMED-LEAF nodes
        for d in range(0, D_star + 1):
            for nid in depth_nodes.get(d, set()):
                nid = int(nid)
                if node_to_top.get(nid, nid) != my_top_node:
                    continue
                if my_child_at_depth.get(d, None) == nid:
                    continue  # skip my own node at this depth

                if is_trimmed_leaf(nid):
                    leaves = node_leaves.get(nid, np.array([], dtype=int))
                    if leaves.size == 1:
                        idx = int(leaves[0])
                        sig = ('leaf', idx)
                        if sig not in seen:
                            seen.add(sig)
                            neg_point_idx.add(idx)
                            negatives_used.append({'depth': d, 'type': 'leaf_point',
                                                   'point_index': idx, 'vector': X[idx]})
                    else:
                        sig = ('leaf_node', nid)
                        if sig not in seen:
                            seen.add(sig)
                            neg_node_ids.add(nid)
                            negatives_used.append({'depth': d, 'type': 'leaf_centroid',
                                                   'node_id': nid, 'vector': node_centroid[nid]})

        # C) Additionally at depth == D*: include NON-LEAF nodes (trimmed sense)
        d = D_star
        for nid in depth_nodes.get(d, set()):
            nid = int(nid)
            if node_to_top.get(nid, nid) != my_top_node:
                continue
            if my_child_at_depth.get(d, None) == nid:
                continue
            if not is_trimmed_leaf(nid):
                sig = ('node', nid)
                if sig not in seen:
                    seen.add(sig)
                    neg_node_ids.add(nid)
                    negatives_used.append({'depth': d, 'type': 'node_centroid',
                                           'node_id': nid, 'vector': node_centroid[nid]})

        # materialize negatives (centroids first, then singleton raw points)
        for nid in sorted(neg_node_ids):
            negatives[i].append(node_centroid[nid])
            # try to fetch lambda at which this node leaves its parent; default to 1.0 if missing
            neg_lambdas[i].append(float(child_lambda.get(nid, 1.0)))
        for idx in sorted(neg_point_idx):
            negatives[i].append(X[idx])
            neg_lambdas[i].append(float(child_lambda.get(idx, 1.0)))

        # attach QA payload
        steps[D_star]['negatives_used'] = negatives_used

    return positives, negatives, lambdas, neg_lambdas, paths
