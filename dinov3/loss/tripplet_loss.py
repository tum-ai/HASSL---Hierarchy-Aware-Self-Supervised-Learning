import torch
import torch.nn as nn
import torch.nn.functional as F
import dinov3.distributed as distributed  # for get_rank()


# TODO: What should we do it no anchor is valid?

class TripletLoss(nn.Module):
    """
    Triplet loss supporting two modes:

    Modes:
      - 'sample' (default): deterministic sample a positive & negative per anchor from the global pool
      - 'batch_hard' : vectorized batch-hard mining across the global pool:
          hardest positive = MIN cosine-similarity among positives (farthest positive)
          hardest negative = MAX cosine-similarity among negatives (closest negative)

    Forward returns (loss_tensor, stats_dict) where stats_dict contains:
      - "valid_count": number of anchors that had valid pos+neg
      - "total_anchors": total local anchors considered
      - "mode": 'sample' or 'batch_hard'
    """
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = float(margin)

    def init_weights(self):
        # API parity with DINOLoss
        return

    def _sample_triplets(self, anchors, global_emb_det, labels_cpu, local_indices_cpu, g):
        """
        Deterministic sampling-based triplets (keeps the logic of your previous implementation).
        returns: loss_tensor, stats_dict
        """
        device = anchors.device
        B_local, D = anchors.shape
        pos_list = []
        neg_list = []
        valid_mask = []

        # build cluster mapping (on CPU) for fast sampling
        cluster_to_indices = {}
        for idx, lab in enumerate(labels_cpu):
            lab = int(lab)
            if lab == -1:
                continue
            cluster_to_indices.setdefault(lab, []).append(idx)

        for local_idx in local_indices_cpu.tolist():
            lab = int(labels_cpu[local_idx])
            if lab == -1:
                valid_mask.append(False)
                pos_list.append(torch.zeros(D, device=device))
                neg_list.append(torch.zeros(D, device=device))
                continue

            candidates = cluster_to_indices.get(lab, [])
            filtered = [c for c in candidates if c != int(local_idx)]
            if len(filtered) == 0:
                valid_mask.append(False)
                pos_list.append(torch.zeros(D, device=device))
                neg_list.append(torch.zeros(D, device=device))
                continue

            # sample positive deterministically
            if len(filtered) == 1:
                pos_idx = filtered[0]
            else:
                ridx = torch.randint(low=0, high=len(filtered), size=(1,), generator=g).item()
                pos_idx = filtered[ridx]

            other_clusters = [c for c in cluster_to_indices.keys() if c != lab]
            if len(other_clusters) == 0:
                valid_mask.append(False)
                pos_list.append(torch.zeros(D, device=device))
                neg_list.append(torch.zeros(D, device=device))
                continue

            if len(other_clusters) == 1:
                chosen_cluster = other_clusters[0]
            else:
                ridx = torch.randint(low=0, high=len(other_clusters), size=(1,), generator=g).item()
                chosen_cluster = other_clusters[ridx]

            neg_candidates = cluster_to_indices[chosen_cluster]
            if len(neg_candidates) == 1:
                neg_idx = neg_candidates[0]
            else:
                ridx = torch.randint(low=0, high=len(neg_candidates), size=(1,), generator=g).item()
                neg_idx = neg_candidates[ridx]

            pos_list.append(global_emb_det[pos_idx].to(device))
            neg_list.append(global_emb_det[neg_idx].to(device))
            valid_mask.append(True)

        if not any(valid_mask):
            return anchors.new_tensor(0.0), {"valid_count": 0, "total_anchors": B_local, "mode": "sample"}

        pos_tensor = torch.stack(pos_list, dim=0)
        neg_tensor = torch.stack(neg_list, dim=0)
        valid_mask_tensor = torch.tensor(valid_mask, dtype=torch.bool, device=device)

        sim_ap = (anchors * pos_tensor).sum(dim=1)
        sim_an = (anchors * neg_tensor).sum(dim=1)
        raw = (sim_an - sim_ap + self.margin).clamp(min=0.0)

        loss_val = raw[valid_mask_tensor].mean()
        return loss_val, {"valid_count": int(valid_mask_tensor.sum().item()), "total_anchors": B_local, "mode": "sample"}

    def _batch_hard(self, anchors, global_emb_det, global_labels, local_indices):
        """
        Vectorized batch-hard triplet mining across the global pool.
        anchors: [B_local, D] normalized
        global_emb_det: [N_total, D] normalized (detached)
        global_labels: LongTensor [N_total]
        local_indices: LongTensor [B_local]
        """
        device = anchors.device
        B_local, D = anchors.shape
        N_total = global_emb_det.shape[0]

        # compute similarities: [B_local, N_total]
        sims = torch.matmul(anchors, global_emb_det.t())  # cosine sims

        # anchor labels drawn from global_labels via local_indices
        anchor_labels = global_labels[local_indices]  # [B_local]

        # build masks
        # positive mask: same label AND not equal to anchor index
        # Note: doing label compare in device tensor
        eq = global_labels.unsqueeze(0) == anchor_labels.unsqueeze(1)  # BxN
        # exclude self
        idx_arange = torch.arange(N_total, device=device).unsqueeze(0).expand(B_local, N_total)
        local_indices_broad = local_indices.unsqueeze(1).expand(B_local, N_total)
        eq = eq & (idx_arange != local_indices_broad)

        neq = global_labels.unsqueeze(0) != anchor_labels.unsqueeze(1)

        # anchors with label == -1 are invalid
        anchor_not_noise = anchor_labels != -1

        # For positives, we need the *hardest* positive = minimum similarity (farthest positive)
        # For negatives, hardest negative = maximum similarity (closest negative)
        # Mask sims accordingly: set non-positives to +inf then take min; non-negatives to -inf then max
        pos_mask = eq
        neg_mask = neq

        # avoid empty-reduction by filling excluded entries
        sims_pos = sims.masked_fill(~pos_mask, float("inf"))  # BxN
        pos_sim, _ = sims_pos.min(dim=1)  # min over N: +inf if no positives

        sims_neg = sims.masked_fill(~neg_mask, float("-inf"))
        neg_sim, _ = sims_neg.max(dim=1)  # -inf if no negatives

        # valid anchors: not noise & have at least one positive and one negative (pos_sim != inf, neg_sim != -inf)
        valid_mask = anchor_not_noise & (pos_sim != float("inf")) & (neg_sim != float("-inf"))

        # compute raw hinge: sim_an - sim_ap + margin -> clamp
        # convert pos_sim/neg_sim to same dtype & device
        pos_sim = pos_sim.to(anchors.dtype)
        neg_sim = neg_sim.to(anchors.dtype)
        raw = (neg_sim - pos_sim + self.margin).clamp(min=0.0)

        valid_count = int(valid_mask.sum().item())
        if valid_count == 0:
            return anchors.new_tensor(0.0), {"valid_count": 0, "total_anchors": B_local, "mode": "batch_hard"}

        loss_val = raw[valid_mask].mean()
        return loss_val, {"valid_count": valid_count, "total_anchors": B_local, "mode": "batch_hard"}

    def forward(
        self,
        anchors: torch.Tensor,
        global_emb: torch.Tensor,
        global_labels: torch.Tensor,
        local_indices: torch.Tensor,
        *,
        margin: float | None = None,
        seed: int | None = None,
        mode: str = "sample",  # "sample" or "batch_hard"
    ) -> tuple[torch.Tensor, dict]:
        if margin is None:
            margin = self.margin

        if anchors is None or global_emb is None or global_labels is None or local_indices is None:
            return anchors.new_tensor(0.0), {"valid_count": 0, "total_anchors": 0, "mode": mode}

        device = anchors.device
        anchors = anchors.contiguous().to(device)
        global_emb = global_emb.contiguous().to(device)
        global_labels = global_labels.contiguous().to(device)
        local_indices = local_indices.contiguous().to(device)

        B_local, D = anchors.shape
        N_total = global_emb.shape[0]
        if B_local == 0 or N_total == 0:
            return anchors.new_tensor(0.0), {"valid_count": 0, "total_anchors": B_local, "mode": mode}

        # normalize anchors and global pool (cosine)
        anchors = F.normalize(anchors, p=2, dim=1)
        global_emb_det = F.normalize(global_emb.detach(), p=2, dim=1)

        # CPU copies for sampling maps if needed
        labels_cpu = global_labels.cpu().long().numpy()
        local_indices_cpu = local_indices.cpu()

        if mode == "batch_hard":
            loss_val, stats = self._batch_hard(anchors, global_emb_det, global_labels, local_indices)
            stats["mode"] = "batch_hard"
            stats["margin"] = float(margin)
            return loss_val, stats

        # else sample-based
        # deterministic CPU generator
        if seed is None:
            rank = distributed.get_rank() if hasattr(distributed, "get_rank") else 0
            seed = int(rank)
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed))

        loss_val, stats = self._sample_triplets(anchors, global_emb_det, labels_cpu, local_indices_cpu, g)
        stats["mode"] = "sample"
        stats["margin"] = float(margin)
        return loss_val, stats
