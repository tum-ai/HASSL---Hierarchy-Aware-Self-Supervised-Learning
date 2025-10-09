import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence, Any
import numpy as np
import os
import logging
import torch

logger = logging.getLogger("dinov3.triplet")


def minmax_scale(x, eps=1e-8):
    t = torch.as_tensor(x, dtype=torch.float32)
    if t.numel() == 0:
        return t  # empty -> empty
    mn, mx = t.min(), t.max()
    return (t - mn) / (mx - mn + eps)

def softmax_weights(x, tau=0.5, eps=1e-6):
    if x.numel() == 0:
        return x
    x = (x - x.mean()) / (x.std(unbiased=False) + eps)
    return F.softmax(x / tau, dim=0)


class TripletHCentroidLoss(nn.Module):
    """
    Triplet loss that consumes HDBSCAN path outputs (`positives`, `negatives`, `lambdas`).
    positives: list[N_total] of lists of centroid vectors (np.ndarray or list) length P_i each -> shape (P_i, D)
    negatives: list[N_total] of lists of centroid vectors (same)
    lambdas: list[N_total] of lists of floats aligned with positives
    neg_lambdas (optional): list[N_total] of lists of floats aligned with negatives
    local_indices: LongTensor (B_local,) indices for anchors in positives/negatives/lambdas lists
    """

    def __init__(
        self,
        margin: float = 0.2,
        weighting_mode: str = "weighted_mean",  # "none" | "weighted_mean" | "weighted_minmax"
        lambda_scaling: Optional[str] = "global",  # "global" | "local" | None
        negative_weighting: str = "uniform",  # "uniform" | "inverse_pos" | "based_on_pos" (simple heuristics)
        eps: float = 1e-8,
    ):
        super().__init__()
        assert weighting_mode in ("none", "weighted_mean", "weighted_minmax")
        assert lambda_scaling in ("global", "local", None)
        assert negative_weighting in ("uniform", "inverse_pos", "based_on_pos", "lambda_mean")
        self.margin = float(margin)
        self.weighting_mode = weighting_mode
        self.lambda_scaling = lambda_scaling
        self.negative_weighting = negative_weighting
        self.eps = float(eps)

    def forward(
        self,
        anchors: torch.Tensor,
        *,
        positives: Sequence[Sequence[Any]],
        negatives: Sequence[Sequence[Any]],
        lambdas: Sequence[Sequence[float]],
        neg_lambdas: Optional[Sequence[Sequence[float]]] = None,
        local_indices: torch.Tensor,
        margin: Optional[float] = None,
    ):
        if margin is None:
            margin = self.margin
        if anchors is None:
            return anchors.new_tensor(0.0), {
                "valid_count": 0,
                "total_anchors": 0,
                "weighting_mode": self.weighting_mode,
                "lambda_scaling": self.lambda_scaling,
                "negative_weighting": self.negative_weighting,
            }

        device = anchors.device
        anchors = anchors.contiguous().to(device)
        # list of indices on CPU for indexing python lists
        local_indices_list = local_indices.contiguous().cpu().tolist()
        B_local, D = anchors.shape

        # Compute quick diagnostics for the batch (counts, lambda stats)
        pos_counts = []
        neg_counts = []
        lambda_lens = []
        lambda_vals_flat = []
        for idx in local_indices_list:
            p = positives[idx] if idx < len(positives) else []
            n = negatives[idx] if idx < len(negatives) else []
            lam = lambdas[idx] if idx < len(lambdas) else []
            pos_counts.append(len(p))
            neg_counts.append(len(n))
            lambda_lens.append(len(lam))
            if lam:
                lambda_vals_flat.extend([float(x) for x in lam])

        pos_counts = np.array(pos_counts, dtype=int)
        neg_counts = np.array(neg_counts, dtype=int)
        lambda_lens = np.array(lambda_lens, dtype=int)

        # Basic logging summary
        logger.debug(f"[TripletH] batch pos_counts: mean={pos_counts.mean() if pos_counts.size else 0:.2f} "
                     f"zeros={(pos_counts==0).sum()}/{len(pos_counts)}")
        logger.debug(f"[TripletH] batch neg_counts: mean={neg_counts.mean() if neg_counts.size else 0:.2f} "
                     f"zeros={(neg_counts==0).sum()}/{len(neg_counts)}")
        if len(lambda_vals_flat) > 0:
            logger.debug(f"[TripletH] lambda stats: min={np.min(lambda_vals_flat):.6f} "
                         f"max={np.max(lambda_vals_flat):.6f} mean={np.mean(lambda_vals_flat):.6f}")
        else:
            logger.debug("[TripletH] lambda stats: no lambda values present")

        # ---- Global min/max for lambda (positives) and neg_lambda (negatives) ----
        '''all_pos_lams = []
        all_neg_lams = []
        for idx in local_indices_list:
            all_pos_lams.extend(lambdas[idx] if idx < len(lambdas) else [])
            if neg_lambdas is not None and idx < len(neg_lambdas):
                all_neg_lams.extend(neg_lambdas[idx])

        # positives
        if len(all_pos_lams) == 0:
            pos_min, pos_max = 0.0, 1.0
        else:
            pos_min = float(np.min(all_pos_lams))
            pos_max = float(np.max(all_pos_lams))
            if abs(pos_max - pos_min) < 1e-12:
                pos_max = pos_min + 1.0

        # negatives
        if len(all_neg_lams) == 0:
            neg_min, neg_max = 0.0, 1.0
        else:
            neg_min = float(np.min(all_neg_lams))
            neg_max = float(np.max(all_neg_lams))
            if abs(neg_max - neg_min) < 1e-12:
                neg_max = neg_min + 1.0
        pos_min_t = torch.tensor(pos_min, device=device, dtype=anchors.dtype)
        pos_max_t = torch.tensor(pos_max, device=device, dtype=anchors.dtype)
        neg_min_t = torch.tensor(neg_min, device=device, dtype=anchors.dtype)
        neg_max_t = torch.tensor(neg_max, device=device, dtype=anchors.dtype)'''
        losses = []
        valid_flags = []

        # MAIN per-anchor loop
        for idx_in_batch, global_idx in enumerate(local_indices_list):
            pos_list = positives[global_idx] if global_idx < len(positives) else []
            neg_list = negatives[global_idx] if global_idx < len(negatives) else []
            lambda_list = lambdas[global_idx] if global_idx < len(lambdas) else []
            neg_lambda_list = (
                neg_lambdas[global_idx] if (neg_lambdas is not None and global_idx < len(neg_lambdas)) else []
            )

            # convert positives -> tensor
            if len(pos_list) == 0:
                valid_flags.append(False)
                continue
            try:
                pos_tensor = torch.tensor(pos_list, dtype=anchors.dtype, device=device)
            except Exception:
                pos_tensor = torch.stack(
                    [torch.as_tensor(p, dtype=anchors.dtype, device=device) for p in pos_list], dim=0
                )

            # convert negatives -> tensor, or try fallback
            if len(neg_list) == 0:
                valid_flags.append(False)
            else:
                try:
                    neg_tensor = torch.tensor(neg_list, dtype=anchors.dtype, device=device)
                except Exception:
                    neg_tensor = torch.stack(
                        [torch.as_tensor(n, dtype=anchors.dtype, device=device) for n in neg_list], dim=0
                    )

            
            # CONTRASTIVE LOSS JR - https://qdrant.tech/articles/triplet-loss/

            #LOCAL SCALING OF LAMBDAS

            lambda_scaled = minmax_scale(lambda_list).to(device=device, dtype=anchors.dtype)
            neg_lambda_scaled = minmax_scale(neg_lambda_list).to(device=device, dtype=anchors.dtype)

            #lambda_t     = torch.as_tensor(lambda_list,     device=device, dtype=anchors.dtype)[:pos_tensor.size(0)]
            #neg_lambda_t = torch.as_tensor(neg_lambda_list, device=device, dtype=anchors.dtype)[:neg_tensor.size(0)]
            
            #w_pos = softmax_weights(lambda_t)         # sums to 1
            #w_neg = softmax_weights(neg_lambda_t)

            #GLOBAL SCALING START OF LAMBDAS

            #lambda_t      = torch.as_tensor(lambda_list,      device=device, dtype=anchors.dtype)
            #neg_lambda_t  = torch.as_tensor(neg_lambda_list,  device=device, dtype=anchors.dtype)

            #lambda_scaled     = (lambda_t     - pos_min_t) / (pos_max_t - pos_min_t + 1e-8)
            #neg_lambda_scaled = (neg_lambda_t - neg_min_t) / (neg_max_t - neg_min_t + 1e-8)


            anchor = F.normalize(anchors[idx_in_batch].float(), p=2, dim=0)   # (D,)
            pos_tensor = F.normalize(pos_tensor.float(),              p=2, dim=1) # (P,D)
            neg_tensor = F.normalize(neg_tensor.float(),              p=2, dim=1) # (N,D)

            # --- POSITIVE SUM (scalar) ---
            positive_sum = anchor.new_tensor(0.0, dtype=torch.float32)
            for i in range(pos_tensor.size(0)):
                # scalar squared L2 to the i-th positive
                d2 = (pos_tensor[i] - anchor).pow(2).sum()                        # scalar
                positive_sum = positive_sum + d2 #* lambda_scaled[i]     NO LAMBDA          # stays scalar

            # --- NEGATIVE SUM (scalar) ---
            negative_sum = anchor.new_tensor(0.0, dtype=torch.float32)
            for i in range(neg_tensor.size(0)):
                # scalar L2 to the i-th negative (unit sphere: max dist ≈ 2)
                d = (neg_tensor[i] - anchor).pow(2).sum().sqrt()                  # scalar
                neg_term = (2 - d).clamp(min=0.0).pow(2) # scalar
                negative_sum = negative_sum + neg_term #* neg_lambda_scaled[i])   NO LAMBDA                        # stays scalar

            '''
            # Normalize
            pos_tensor = F.normalize(pos_tensor, p=2, dim=1)
            neg_tensor = F.normalize(neg_tensor, p=2, dim=1)
            anchor = anchors[idx_in_batch].unsqueeze(1)  # (D, 1)

            sim_pos = (pos_tensor @ anchor).squeeze(1)  # (P,)
            sim_neg = (neg_tensor @ anchor).squeeze(1)  # (Nneg,)

            # weights from lambda_list
            if len(lambda_list) == 0:
                pos_w = torch.ones((sim_pos.shape[0],), dtype=anchors.dtype, device=device)
            else:
                pos_w = torch.tensor(lambda_list, dtype=anchors.dtype, device=device)
                if self.lambda_scaling == "local":
                    if pos_w.numel() == 0:
                        pos_w = torch.ones_like(pos_w)
                    else:
                        wmin = float(pos_w.min().detach().cpu().item())
                        wmax = float(pos_w.max().detach().cpu().item())
                        if abs(wmax - wmin) < 1e-12:
                            pos_w = torch.ones_like(pos_w)
                        else:
                            pos_w = ((pos_w - wmin) / (wmax - wmin)).clamp(min=self.eps)
                elif self.lambda_scaling == "global":
                    pos_w = ((pos_w - global_min) / (global_max - global_min)).clamp(min=self.eps)
                else:
                    pos_w = pos_w.clamp(min=self.eps)

            # negative weighting heuristics / from provided negative lambdas
            if self.negative_weighting == "uniform":
                neg_w = torch.ones_like(sim_neg)
            elif self.negative_weighting == "inverse_pos":
                inv = 1.0 / (1.0 + float(pos_w.mean().detach().cpu().item()))
                neg_w = torch.full_like(sim_neg, fill_value=inv)
            elif self.negative_weighting == "based_on_pos":
                tau = 0.1
                scaled = sim_neg / tau
                neg_w = torch.softmax(scaled, dim=0)
            else:  # lambda_mean
                if len(neg_lambda_list) == 0:
                    neg_w = torch.ones_like(sim_neg)
                else:
                    try:
                        neg_w = torch.tensor(neg_lambda_list, dtype=anchors.dtype, device=device)
                    except Exception:
                        neg_w = torch.stack(
                            [torch.as_tensor(v, dtype=anchors.dtype, device=device) for v in neg_lambda_list], dim=0
                        )
                    neg_w = neg_w.clamp(min=self.eps)

            # selection
            if self.weighting_mode == "none":
                sim_ap = sim_pos.min()
                if self.negative_weighting == "uniform":
                    # Use all negatives with uniform weight (mean over negatives)
                    sim_an = sim_neg.mean()
                elif self.negative_weighting == "lambda_mean":
                    # Weighted mean with provided negative lambdas (defaults to ones)
                    neg_w_norm = neg_w / (neg_w.sum() + self.eps)
                    sim_an = (sim_neg * neg_w_norm).sum()
                else:
                    sim_an = sim_neg.max()
            elif self.weighting_mode == "weighted_mean":
                pos_w_norm = pos_w / (pos_w.sum() + self.eps)
                sim_ap = (sim_pos * pos_w_norm).sum()
                if self.negative_weighting == "uniform":
                    # Use all negatives with uniform weight (mean over negatives)
                    sim_an = sim_neg.mean()
                elif self.negative_weighting == "lambda_mean":
                    # Weighted mean with provided negative lambdas (defaults to ones)
                    neg_w_norm = neg_w / (neg_w.sum() + self.eps)
                    sim_an = (sim_neg * neg_w_norm).sum()
                else:
                    sim_an = (sim_neg * neg_w).max()
            elif self.weighting_mode == "weighted_minmax":
                adj_pos = sim_pos / (pos_w + self.eps)
                sim_ap = adj_pos.min()
                if self.negative_weighting == "uniform":
                    # Use all negatives with uniform weight (mean over negatives)
                    sim_an = sim_neg.mean()
                elif self.negative_weighting == "lambda_mean":
                    # Weighted mean with provided negative lambdas (defaults to ones)
                    neg_w_norm = neg_w / (neg_w.sum() + self.eps)
                    sim_an = (sim_neg * neg_w_norm).sum()
                else:
                    sim_an = (sim_neg * neg_w).max()
            else:
                sim_ap = sim_pos.min()
                if self.negative_weighting == "uniform":
                    # Use all negatives with uniform weight (mean over negatives)
                    sim_an = sim_neg.mean()
                elif self.negative_weighting == "lambda_mean":
                    # Weighted mean with provided negative lambdas (defaults to ones)
                    neg_w_norm = neg_w / (neg_w.sum() + self.eps)
                    sim_an = (sim_neg * neg_w_norm).sum()
                else:
                    sim_an = sim_neg.max()'''

            #raw = (sim_an - sim_ap + float(margin)).clamp(min=0.0)
            #print(f'positive_sum {positive_sum}')
            #print(f'negative_sum {negative_sum}')


            # combine (optionally average by counts)
            raw = positive_sum / max(1, pos_tensor.size(0)) + negative_sum / max(1, neg_tensor.size(0))
            print(f'Positive sum: {positive_sum}')
            print(f'Negative sum: {negative_sum}')
            print(f'Raw: {raw}')
            losses.append(raw)
            valid_flags.append(True)

        # If nothing valid -> large debug dump and return zero loss (but with diagnostics)
        if not any(valid_flags):
            # verbose logging
            logger.warning(
                f"[Triplet HDBScan] valid_anchors=0/{B_local} "
                f"pos_zero_fraction={(pos_counts==0).sum()}/{len(pos_counts)} "
                f"neg_zero_fraction={(neg_counts==0).sum()}/{len(neg_counts)}"
            )

            return anchors.new_tensor(0.0), {
                "valid_count": 0,
                "total_anchors": B_local,
                "weighting_mode": self.weighting_mode,
                "lambda_scaling": self.lambda_scaling,
                "negative_weighting": self.negative_weighting,
            }

        # Check if individual losses maintain gradients
        valid_losses = [l for l, v in zip(losses, valid_flags) if v]
        if valid_losses:
            # Check gradient connectivity of individual losses
            first_loss = valid_losses[0]
            logger.debug(f"First valid loss requires_grad: {first_loss.requires_grad}, grad_fn: {first_loss.grad_fn}")
        
        loss_tensor = torch.stack(valid_losses).mean()
        valid_count = int(sum(1 for v in valid_flags if v))
        stats = {
            "valid_count": valid_count,
            "total_anchors": B_local,
            "weighting_mode": self.weighting_mode,
            "lambda_scaling": self.lambda_scaling,
            "negative_weighting": self.negative_weighting,
        }
        return loss_tensor, stats