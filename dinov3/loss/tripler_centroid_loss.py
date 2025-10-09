import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletCentroidLoss(nn.Module):
    """
    Centroid-based triplet loss.

    For each anchor we:
      - collect positive centroids = centroids of the clusters the anchor belongs to across all eps levels
      - collect negative centroids = all other centroids (all eps) excluding the positive centroids
      - compute cosine similarities between anchor and positives/negatives
      - hardened formulation: sim_ap = min(sim_pos), sim_an = max(sim_neg)
      - loss_i = max(0, sim_an - sim_ap + margin)
    Average over anchors with at least one positive and one negative.

    Inputs:
      anchors: [B_local, D] (student embeddings, requires grad)
      centroids_per_eps: list of L tensors, each (n_clusters_j, D) (already normalized or will be normalized here)
      centroid_labels_per_eps: list of L 1D tensors with the corresponding cluster labels for each centroid row
      labels_per_eps: LongTensor (L, N_total) with DBSCAN labels per eps for the full pooled dataset
      local_indices: LongTensor (B_local,) indices into N_total for this rank
    """
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = float(margin)

    def init_weights(self):
        # Staying consistent with api
        return

    def forward(
        self,
        anchors: torch.Tensor,
        centroids_per_eps: list,
        centroid_labels_per_eps: list,
        labels_per_eps: torch.Tensor,
        local_indices: torch.Tensor,
        *,
        margin: float | None = None,
    ):
        if margin is None:
            margin = self.margin

        if anchors is None or centroids_per_eps is None or centroid_labels_per_eps is None or labels_per_eps is None or local_indices is None:
            return anchors.new_tensor(0.0), {"valid_count": 0, "total_anchors": 0}

        device = anchors.device
        anchors = anchors.contiguous().to(device)
        labels_per_eps = labels_per_eps.contiguous().to(device)
        local_indices = local_indices.contiguous().to(device)

        B_local, D = anchors.shape
        L = labels_per_eps.shape[0]  # number of eps levels

        # normalize anchors and centroids
        anchors = F.normalize(anchors, p=2, dim=1)
        # centroids_per_eps are expected to already be normalized in get_clustering;
        # but normalize again to be safe (no grad for centroids)
        centroids_normed = [F.normalize(c.detach(), p=2, dim=1) if c.numel() else c for c in centroids_per_eps]

        # Precompute flattened negatives arrays (all centroids concatenated) and bookkeeping to quickly exclude positives
        flat_centroids = []
        flat_labels = []  # tuples (eps_idx, centroid_label)
        for eps_i, (centroids_i, labels_i) in enumerate(zip(centroids_normed, centroid_labels_per_eps)):
            if centroids_i.numel() == 0:
                continue
            flat_centroids.append(centroids_i)  # (n_i, D)
            # store pair (eps_index, cluster_label as int)
            flat_labels.extend([(eps_i, int(x.item())) for x in labels_i])
        if len(flat_centroids) == 0:
            return anchors.new_tensor(0.0), {"valid_count": 0, "total_anchors": B_local}
        flat_centroids = torch.cat(flat_centroids, dim=0)  # (M_total, D)
        # flat_labels is Python list length M_total

        losses = []
        valid_flags = []
        # per-anchor loop: small (B_local ~ 64-256). This is fine and simple to reason about.
        labels_per_eps_cpu = labels_per_eps.cpu().numpy()  # (L, N_total) as numpy for fast indexing
        for idx_in_batch, global_idx in enumerate(local_indices.cpu().tolist()):
            # collect positives across eps: for each eps, get label for this anchor
            positives = []
            for eps_i in range(L):
                lab = int(labels_per_eps_cpu[eps_i, global_idx])
                if lab == -1:
                    continue
                # find centroid index in flat list corresponding to (eps_i, lab)
                # linear scan (M_total small) is OK; save consistent mapping if needed
                for centroid_idx, (eps_j, clab_j) in enumerate(flat_labels):
                    if eps_j == eps_i and clab_j == lab:
                        positives.append(flat_centroids[centroid_idx].to(device))
                        break
            if len(positives) == 0:
                valid_flags.append(False)
                continue

            # negatives = all centroids except those exactly in positives
            # build a mask over flat_centroids to exclude exact matches
            # Because centroids are unique per (eps,label) we can identify by flat_labels
            # build positive index set by matching labels_per_eps anchor labels
            pos_idx_set = set()
            anchor_label_tuple_set = set()
            for eps_i in range(L):
                lab = int(labels_per_eps_cpu[eps_i, global_idx])
                if lab == -1:
                    continue
                anchor_label_tuple_set.add((eps_i, lab))
            for centroid_idx, (eps_j, clab_j) in enumerate(flat_labels):
                if (eps_j, clab_j) in anchor_label_tuple_set:
                    pos_idx_set.add(centroid_idx)
            # negatives are centroid indices not in pos_idx_set
            neg_idx_list = [i for i in range(flat_centroids.shape[0]) if i not in pos_idx_set]
            if len(neg_idx_list) == 0:
                valid_flags.append(False)
                continue

            # create tensors
            pos_tensor = torch.stack(positives, dim=0)  # (P, D)
            neg_tensor = flat_centroids[neg_idx_list].to(device)  # (Nneg, D)

            anchor = anchors[idx_in_batch].unsqueeze(0)  # (1, D)
            # cosine similarities
            if pos_tensor.dtype != anchor.dtype:
                pos_tensor = pos_tensor.to(dtype=anchor.dtype, device=anchor.device)
            sim_pos = torch.matmul(pos_tensor, anchor.t()).squeeze(1)  # (P,)
            if neg_tensor.dtype != anchor.dtype:
                neg_tensor = neg_tensor.to(dtype=anchor.dtype, device=anchor.device)
            sim_neg = torch.matmul(neg_tensor, anchor.t()).squeeze(1)  # (Nneg,)
            # hardest positive = min(sim_pos), hardest negative = max(sim_neg)
            sim_ap = sim_pos.min()
            sim_an = sim_neg.max()

            raw = (sim_an - sim_ap + float(margin)).clamp(min=0.0)
            losses.append(raw)
            valid_flags.append(True)

        if not any(valid_flags):
            return anchors.new_tensor(0.0), {"valid_count": 0, "total_anchors": B_local}
        selected_losses = [l for l, v in zip(losses, valid_flags) if v]
        if len(selected_losses) == 0:
            if len(losses) > 0:
                ref = losses[0]
            else:
                # fallback: use anchor as reference
                ref = anchor if 'anchor' in locals() else torch.tensor(0.0)

            zero = ref.new_tensor(0.0, requires_grad=True)
            loss_tensor = zero
            triplet_stats = {
                "num_valid_triplets": 0,
                "num_candidate_triplets": len(losses),
            }
            return loss_tensor, triplet_stats
        loss_tensor = torch.stack([l for l, v in zip(losses, valid_flags) if v]).mean()
        valid_count = int(sum(1 for v in valid_flags if v))
        return loss_tensor, {"valid_count": valid_count, "total_anchors": B_local}
