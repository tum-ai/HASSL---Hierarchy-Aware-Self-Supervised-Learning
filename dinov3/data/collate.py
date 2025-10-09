# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import random
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

TARGET_SIZE = 224

def _to_tensor_chw(x) -> torch.Tensor:
    """
    Convert x to a torch tensor in CHW format (C,H,W), float32.
    Accepts: torch.Tensor (CHW/HWC), numpy arrays (HWC or HW), PIL images (handled by torchvision.ToTensor earlier).
    """
    if isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.as_tensor(x)

    # If HWC (3 dims and last dim small and not channels-first) -> convert
    if t.ndim == 3:
        # Heuristic: if last dim looks like channels (<=4) and first dims are large -> probably HWC
        if t.shape[2] <= 4 and t.shape[0] > 4:
            # HWC -> CHW
            t = t.permute(2, 0, 1)
        # else assume already CHW
    elif t.ndim == 2:
        # single channel HW -> add channel dim
        t = t.unsqueeze(0)
    elif t.ndim == 4:
        # if somebody passed a batch tensor, take first element (shouldn't normally happen)
        t = t[0]
    else:
        raise RuntimeError(f"Unsupported image tensor ndim={t.ndim}")

    # ensure float32 for interpolation / processing
    if not t.is_floating_point():
        t = t.float()
    else:
        if t.dtype != torch.float32:
            t = t.to(torch.float32)

    return t.contiguous()


def _normalize_channels(t: torch.Tensor) -> torch.Tensor:
    """
    Ensure 3 channels. If 1 -> repeat; if >3 -> take first 3.
    Input expected CHW.
    """
    c, h, w = t.shape
    if c == 1:
        t = t.repeat(3, 1, 1)
    elif c >= 3:
        if c > 3:
            t = t[:3, :, :]
    else:
        # shouldn't happen, but handle it
        t = t.repeat(3 // c + 1, 1, 1)[:3, :, :]
    return t


def _stack_and_resize(tensors: Sequence, target_size: Optional[Union[int, Tuple[int, int]]] = None):
    """
    Given sequence of raw tensors/arrays, returns a stacked tensor shape (N, C, H, W).
    If target_size is None -> compute max H,W among tensors and use that.
    target_size can be int or (H,W).
    """
    proc = []
    heights = []
    widths = []
    for x in tensors:
        t = _to_tensor_chw(x)
        t = _normalize_channels(t)
        proc.append(t)
        heights.append(t.shape[1])
        widths.append(t.shape[2])

    if target_size is None:
        H = int(max(heights))
        W = int(max(widths))
    else:
        if isinstance(target_size, int):
            H = W = int(target_size)
        else:
            H, W = target_size

    resized = []
    for t in proc:
        if t.shape[1] == H and t.shape[2] == W:
            resized.append(t)
        else:
            t4 = t.unsqueeze(0)  # 1,C,H0,W0
            # interpolate expects float tensors in BCHW
            t_res = F.interpolate(t4, size=(H, W), mode="bilinear", align_corners=False)
            resized.append(t_res.squeeze(0))
    return torch.stack(resized, dim=0)  # N, C, H, W


def collate_data_and_cast(
    samples_list,
    mask_ratio_tuple,
    mask_probability,
    dtype,
    n_tokens=None,
    mask_generator=None,
    random_circular_shift=False,
    local_batch_size=None,
):
    n_global_crops = len(samples_list[0][0]["global_crops"])
    n_local_crops = len(samples_list[0][0]["local_crops"])

    collated_global_crops = torch.stack(
        [s[0]["global_crops"][i] for i in range(n_global_crops) for s in samples_list]
    )  # [n_global_crops, B, ...]
    collated_local_crops = torch.stack([s[0]["local_crops"][i] for i in range(n_local_crops) for s in samples_list])
    if "gram_teacher_crops" in samples_list[0][0]:
        collated_gram_teacher_crops = torch.stack(
            [s[0]["gram_teacher_crops"][i] for i in range(n_global_crops) for s in samples_list]
        )  # [n_global_crops, B, ...]
    else:
        collated_gram_teacher_crops = None

    if "image" in samples_list[0][0]:
        images = [s[0]["image"] for s in samples_list]
        collate_images = _stack_and_resize(images, target_size=TARGET_SIZE)
    else:
        collate_images = None

    if local_batch_size is not None:
        # multi-distillation case, number of masks is different because the number of samples masked
        # is different of the number of samples passed into the teacher initially
        B = n_global_crops * local_batch_size
    else:
        B = len(collated_global_crops)
    N = n_tokens
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    masks_list = []
    for i in range(0, n_samples_masked):
        prob_max = probs[i + 1]
        mask = torch.BoolTensor(mask_generator(int(N * prob_max)))
        if random_circular_shift:  # apply le random circular shift to
            shift_x, shift_y = (
                random.randint(0, mask.shape[0] - 1),
                random.randint(0, mask.shape[1] - 1),
            )
            mask = torch.roll(mask, (shift_x, shift_y), (0, 1))
        masks_list.append(mask)
        upperbound += int(N * prob_max)
    for _ in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)

    collated_masks = torch.stack(masks_list).flatten(1)
    mask_indices_list = collated_masks.flatten().nonzero().flatten()

    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

    out = {
        "collated_global_crops": collated_global_crops.to(dtype),
        "collated_local_crops": collated_local_crops.to(dtype),
        "collated_images": collate_images.to(dtype),
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
    }
    if collated_gram_teacher_crops is not None:
        out["collated_gram_teacher_crops"] = collated_gram_teacher_crops.to(dtype)
    return out


# def get_batch_subset(collated_data_batch, target_bs):
def get_batch_subset(collated_data_batch, divide_by):
    old_bs = collated_data_batch["collated_global_crops"].shape[0] // 2
    target_bs = (old_bs + divide_by - 1) // divide_by
    collated_global_crops = (
        collated_data_batch["collated_global_crops"].unflatten(0, (2, old_bs)).narrow(1, 0, target_bs).flatten(0, 1)
    )
    collated_local_crops = (
        collated_data_batch["collated_local_crops"].unflatten(0, (-1, old_bs)).narrow(1, 0, target_bs).flatten(0, 1)
    )

    masks_old_bs = collated_data_batch["collated_masks"].shape[0] // 2
    masks_target_bs = masks_old_bs // divide_by
    collated_masks = (
        collated_data_batch["collated_masks"]
        .unflatten(0, (2, masks_old_bs))
        .narrow(1, 0, masks_target_bs)
        .flatten(0, 1)
    )
    mask_indices_list = collated_masks.flatten().nonzero().flatten()

    while mask_indices_list.shape[0] == 0:
        _unbind = list(collated_data_batch["collated_masks"].unbind(0))
        random.shuffle(_unbind)
        _bind = torch.stack(_unbind, dim=0)
        collated_masks = _bind.unflatten(0, (2, masks_old_bs)).narrow(1, 0, masks_target_bs).flatten(0, 1)
        mask_indices_list = collated_masks.flatten().nonzero().flatten()

    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]
    upperbound = collated_data_batch["upperbound"]

    new_batch = {
        "collated_global_crops": collated_global_crops,
        "collated_local_crops": collated_local_crops,
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
    }

    if "global_batch_size" in collated_data_batch.keys():
        new_batch["global_batch_size"] = collated_data_batch["global_batch_size"] // divide_by

    return new_batch
