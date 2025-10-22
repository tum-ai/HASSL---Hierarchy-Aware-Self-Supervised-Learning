# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import gc
import logging
from functools import partial
from pathlib import Path
import numpy as np

import torch
from omegaconf import OmegaConf
from torch import Tensor, nn
import hdbscan
import warnings

# Silence that specific sklearn FutureWarning coming from older name `force_all_finite`.
warnings.filterwarnings(
    "ignore",
    message=r".*force_all_finite.*",
    category=FutureWarning,
)

import dinov3.distributed as distributed
from dinov3.checkpointer import init_fsdp_model_from_checkpoint
from dinov3.configs import get_default_config
from dinov3.data import DataAugmentationDINO
from dinov3.fsdp.ac_compile_parallelize import ac_compile_parallelize
from dinov3.layers.dino_head import DINOHead
from dinov3.loss import DINOLoss, GramLoss, KoLeoLoss, KoLeoLossDistributed, iBOTPatchLoss, TripletLoss, TripletCentroidLoss, TripletHCentroidLoss
from dinov3.models import build_model_from_cfg
from dinov3.train.cosine_lr_scheduler import linear_warmup_cosine_decay
from dinov3.train.param_groups import fuse_params_groups, get_params_groups_with_decay_fsdp
from dinov3.utils import count_parameters
from dinov3.layers.clustering import build_triplet_lists_with_paths

logger = logging.getLogger("dinov3")


class SSLMetaArch(nn.Module):
    """
    Modified version of SSLMetaArchCompilable including gram loss:
    - Gram loss is used only if gram.use_loss is set to true
    """

    def __init__(self, cfg):
        super().__init__()

        # assert cfg.multidistillation.enabled is False
        assert cfg.crops.local_crops_number > 0
        assert cfg.ibot.separate_head is True
        assert cfg.train.centering == "sinkhorn_knopp"

        # For some reason FULL_SHARD doesn't work
        assert cfg.compute_precision.sharding_strategy == "SHARD_GRAD_OP"

        self.cfg = cfg

        student_model_dict = dict()
        teacher_model_dict = dict()
        gram_model_dict = dict()

        student_backbone, teacher_backbone, embed_dim = build_model_from_cfg(cfg)
        torch.cuda.empty_cache()
        gc.collect()
        gram_backbone, _ = build_model_from_cfg(cfg, only_teacher=True)
        logger.info(f"Number of parameters: {count_parameters(student_backbone)}")
        student_model_dict["backbone"] = student_backbone
        teacher_model_dict["backbone"] = teacher_backbone
        gram_model_dict["backbone"] = gram_backbone
        logger.info(f"OPTIONS -- architecture : embed_dim: {embed_dim}")

        self.embed_dim = embed_dim  # D
        self.dino_out_dim = cfg.dino.head_n_prototypes  # K

        logger.info("OPTIONS -- DINO")
        logger.info(f"OPTIONS -- DINO -- loss_weight: {cfg.dino.loss_weight}")
        logger.info(f"OPTIONS -- DINO -- global_ignore_diagonal: {cfg.dino.global_ignore_diagonal}")
        logger.info(f"OPTIONS -- DINO -- head_n_prototypes: {cfg.dino.head_n_prototypes}")
        logger.info(f"OPTIONS -- DINO -- head_bottleneck_dim: {cfg.dino.head_bottleneck_dim}")
        logger.info(f"OPTIONS -- DINO -- head_hidden_dim: {cfg.dino.head_hidden_dim}")
        logger.info(f"OPTIONS -- DINO -- head_norm_last_layer: {cfg.dino.head_norm_last_layer}")
        dino_head_class = partial(
            DINOHead,
            in_dim=embed_dim,
            out_dim=cfg.dino.head_n_prototypes,
            hidden_dim=cfg.dino.head_hidden_dim,
            bottleneck_dim=cfg.dino.head_bottleneck_dim,
            nlayers=cfg.dino.head_nlayers,
        )
        student_model_dict["dino_head"] = dino_head_class()
        teacher_model_dict["dino_head"] = dino_head_class()
        self.dino_loss = DINOLoss(self.dino_out_dim)
        if self.cfg.get("triplet", None) and self.cfg.triplet.enabled:
            self.triplet_loss = TripletHCentroidLoss(
                margin=self.cfg.triplet.get("margin", 0.2),
                weighting_mode=self.cfg.triplet.get("weighting_mode", "weighted_mean"),
                lambda_scaling="global",  # "global" | "local" | None
                negative_weighting="uniform",  # "uniform" | "inverse_pos" | "based_on_pos" (simple heuristics)
                eps=1e-8,
            )
            # self.triplet_centroid_loss = TripletCentroidLoss(self.cfg.triplet.get("margin", 0.2))
            # self.triplet_loss = TripletLoss(self.cfg.triplet.get("margin", 0.2))
        else:
            self.triplet_loss = None

        logger.info("OPTIONS -- KOLEO")
        logger.info(f"OPTIONS -- KOLEO -- loss_weight: {cfg.dino.koleo_loss_weight}")
        logger.info(f"OPTIONS -- KOLEO -- distributed: {cfg.dino.koleo_loss_distributed}")
        if cfg.dino.koleo_loss_distributed:
            logger.info(f"OPTIONS -- KOLEO -- topk: {cfg.dino.koleo_topk}")
            logger.info(
                f"OPTIONS -- KOLEO -- distributed_loss_group_size: {cfg.dino.koleo_distributed_loss_group_size}"
            )
            assert cfg.dino.koleo_distributed_replicas == 0, (
                "Option `dino.koleo_distributed_replicas` is no longer supported"
            )
            self.koleo_loss = KoLeoLossDistributed(
                topk=cfg.dino.koleo_topk,
                loss_group_size=cfg.dino.koleo_distributed_loss_group_size,
            )
        else:
            assert cfg.dino.koleo_topk == 1, "Non-distributed KoLeo loss only supports `dino.koleo_topk=1`"
            self.koleo_loss = KoLeoLoss()

        logger.info("OPTIONS -- IBOT")
        logger.info(f"OPTIONS -- IBOT -- loss_weight: {cfg.ibot.loss_weight}")
        logger.info(f"OPTIONS -- IBOT masking -- ibot_mask_ratio_tuple: {cfg.ibot.mask_ratio_min_max}")
        logger.info(f"OPTIONS -- IBOT masking -- ibot_mask_sample_probability: {cfg.ibot.mask_sample_probability}")

        assert 0 <= cfg.ibot.mask_ratio_min_max[0] < cfg.ibot.mask_ratio_min_max[1] <= 1, (
            "provide a valid cfg.ibot.mask_ratio_min_max"
        )
        assert 0 <= cfg.ibot.mask_sample_probability <= 1, "provide a positive mask probability for ibot"
        logger.info(f"OPTIONS -- IBOT -- head_n_prototypes: {cfg.ibot.head_n_prototypes}")
        logger.info(f"OPTIONS -- IBOT -- head_bottleneck_dim: {cfg.ibot.head_bottleneck_dim}")
        logger.info(f"OPTIONS -- IBOT -- head_hidden_dim: {cfg.ibot.head_hidden_dim}")
        logger.info(f"OPTIONS -- IBOT -- head_norm_last_layer: {cfg.ibot.head_norm_last_layer}")
        ibot_head_class = partial(
            DINOHead,
            in_dim=embed_dim,
            out_dim=cfg.ibot.head_n_prototypes,
            hidden_dim=cfg.ibot.head_hidden_dim,
            bottleneck_dim=cfg.ibot.head_bottleneck_dim,
            nlayers=cfg.ibot.head_nlayers,
        )
        student_model_dict["ibot_head"] = ibot_head_class()
        teacher_model_dict["ibot_head"] = ibot_head_class()
        self.ibot_patch_loss = iBOTPatchLoss(cfg.ibot.head_n_prototypes)

        # Build student and teacher models
        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)
        self.model_ema = self.teacher  # this may be overwritten for distillation
        logger.info(f"Student and Teacher are built: they are both {cfg.student.arch} network.")

        if cfg.distillation.enabled:
            self._setup_distillation()
        # No grad is needed for these two
        self.teacher.requires_grad_(False)
        self.model_ema.requires_grad_(False)
        self.ema_params_lists = None

        # getting config params fixed:
        self.n_local_crops = self.cfg.crops.local_crops_number
        self.is_distillation_enabled = self.cfg.distillation.enabled
        self.dino_global_ignore_diagonal = self.cfg.dino.global_ignore_diagonal
        self.dino_loss_weight = self.cfg.dino.loss_weight
        self.dino_koleo_loss_weight = self.cfg.dino.koleo_loss_weight
        self.ibot_loss_weight = self.cfg.ibot.loss_weight

        # Local loss reweighting
        if self.cfg.dino.reweight_dino_local_loss:
            iter_per_epoch = cfg.train.OFFICIAL_EPOCH_LENGTH
            total_iterations = iter_per_epoch * cfg.optim.epochs
            schedule_cfg = cfg.dino.local_loss_weight_schedule
            self.dino_local_loss_schedule = linear_warmup_cosine_decay(
                start=schedule_cfg.start,
                peak=schedule_cfg.peak,
                end=schedule_cfg.end,
                warmup_iterations=iter_per_epoch * schedule_cfg.warmup_epochs,
                total_iterations=total_iterations,
                cosine_iterations=(
                    iter_per_epoch * schedule_cfg.cosine_epochs if "cosine_epochs" in schedule_cfg else None
                ),
            )
        # triplet
        self.use_triplet_loss = self.cfg.triplet.enabled
        # Gram
        self.gram_use_loss = self.cfg.gram.use_loss
        self.gram_ema_teacher = False
        self.has_gram_teacher = False
        self.gram_teacher_initialized = False
        if self.gram_use_loss:
            # Gram regularization
            self.gram_loss = GramLoss(
                apply_norm=self.cfg.gram.normalized,
                remove_only_teacher_neg=self.cfg.gram.remove_only_teacher_neg,
                remove_neg=self.cfg.gram.remove_neg,
            )
            # Construct gram teacher
            self.has_gram_teacher = True if not cfg.gram.ema_teacher else False
            if self.has_gram_teacher:
                self.gram_teacher = nn.ModuleDict(gram_model_dict)
                self.gram_teacher.requires_grad_(False)
                logger.info(f"Gram teacher parameter at init: {next(self.gram_teacher.named_parameters())}")
            else:
                self.gram_teacher = None

            self.gram_loss_weight = self.cfg.gram.loss_weight
            if self.cfg.gram.get("loss_weight_schedule"):
                iter_per_epoch = cfg.train.OFFICIAL_EPOCH_LENGTH
                total_iterations = iter_per_epoch * cfg.optim.epochs
                schedule_cfg = self.cfg.gram.loss_weight_schedule
                self.gram_loss_schedule = linear_warmup_cosine_decay(
                    start=schedule_cfg.start,
                    peak=schedule_cfg.peak,
                    end=schedule_cfg.end,
                    warmup_iterations=iter_per_epoch * schedule_cfg.warmup_epochs,
                    total_iterations=total_iterations,
                    cosine_iterations=(
                        iter_per_epoch * schedule_cfg.cosine_epochs if "cosine_epochs" in schedule_cfg else None
                    ),
                )
                logger.info(f"Applying gram loss weight schedule instead of `cfg.gram.loss_weight`: {schedule_cfg}")
            else:
                self.gram_loss_schedule = None
            self.gram_ema_teacher = self.cfg.gram.ema_teacher  # If true use the EMA_teacher as gram_teacher
            self.gram_ckpt = self.cfg.gram.ckpt  # Checkpoint to the first gram teacher model
            self.gram_img_level = self.cfg.gram.img_level  # Apply the loss on the image, if false on the batch
            self.gram_tokens_used = self.cfg.gram.tokens_used  # Any value in ["all", "masked", "unmasked"]
            # Update the teacher frequently
            self.gram_rep_update = self.cfg.gram.rep_update  # bool, if yes the gram teacher will be updated at the freq
            self.gram_update_frequency = self.cfg.gram.update_frequency  # defined by this var update_frequency
            self.gram_it_first_update = self.cfg.gram.it_first_update  # after iteration it_first_update is passed.
            self.gram_it_load_ema_teacher = (
                self.cfg.gram.it_load_ema_teacher
            )  # after iteration it_load_ema the ema teacher is loaded into the gram teacher
            self.gram_compute_stats = self.cfg.gram.compute_stats  # whether to compute auxiliary stats
            self.gram_params_lists = None

            if self.gram_ema_teacher and self.gram_ckpt is not None:
                raise ValueError(
                    "Cannot use both `gram.ema_teacher` and `gram.ckpt` at the same time. Please set one of them to False."
                )
            if self.gram_ckpt is None and self.gram_it_load_ema_teacher < 0:
                raise ValueError(
                    "If no gram checkpoint is provided, `gram.it_load_ema_teacher` must be set to a non-negative value."
                )

            assert not (self.gram_ema_teacher and self.gram_rep_update)
            assert self.gram_tokens_used in ["all", "masked", "unmasked"]
            # Currently using masked/unmasked not handle at the image-level
            if self.gram_tokens_used in ["masked", "unmasked"]:
                assert self.gram_img_level is False

            logger.info("OPTIONS -- GRAM")
            logger.info(f"OPTIONS -- GRAM -- loss_weight: {cfg.gram.loss_weight}")
            logger.info(f"OPTIONS -- GRAM -- ema teacher: {cfg.gram.ema_teacher}")
            logger.info(f"OPTIONS -- GRAM -- ckpt: {cfg.gram.ckpt}")
            if self.cfg.gram.rep_update:
                logger.info(f"OPTIONS -- GRAM -- repeated update: {cfg.gram.rep_update}")
                logger.info(f"OPTIONS -- GRAM -- update freq: {cfg.gram.update_frequency}")
                logger.info(f"OPTIONS -- GRAM -- iteration first update: {cfg.gram.it_first_update}")

            logger.info(f"OPTIONS -- GRAM -- tokens_used: {cfg.gram.tokens_used}")
            logger.info(f"OPTIONS -- GRAM -- apply normalization: {cfg.gram.normalized}")
            logger.info(f"OPTIONS -- GRAM -- img_level: {cfg.gram.img_level}")
            logger.info(f"OPTIONS -- GRAM -- remove_neg: {cfg.gram.remove_neg}")
            logger.info(f"OPTIONS -- GRAM -- remove_only_teacher_neg: {cfg.gram.remove_only_teacher_neg}")

            if cfg.crops.gram_teacher_crops_size is None and self.has_gram_teacher:
                raise ValueError("cfg.crops.gram_teacher_crops_size must be set to use gram loss")
            if cfg.crops.gram_teacher_crops_size is not None and self.gram_ema_teacher:
                raise ValueError("cfg.crops.gram_teacher_crops_size shoud be None when gram.ema_teacher=True")

            self.student_crop_size = cfg.crops.global_crops_size
            self.gram_global_teacher_resize_method = cfg.gram.global_teacher_resize_method
            self.gram_global_teacher_resize_antialias = cfg.gram.global_teacher_resize_antialias
            logger.info(f"OPTIONS -- global crops student/teacher size: {self.student_crop_size}")
            logger.info(f"OPTIONS -- global crops GRAM teacher size: {cfg.crops.gram_teacher_crops_size}")
            logger.info(f"OPTIONS -- global crops GRAM teacher resize method: {cfg.gram.global_teacher_resize_method}")
            logger.info(
                f"OPTIONS -- global crops GRAM teacher resize antialias: {cfg.gram.global_teacher_resize_antialias}"
            )

    def _setup_distillation(self):
        logger.info(f"Performing distillation from {self.cfg.distillation.full_cfg_path}")

        default_cfg = get_default_config()
        distillation_cfg = OmegaConf.load(self.cfg.distillation.full_cfg_path)
        distillation_cfg = OmegaConf.merge(default_cfg, distillation_cfg)

        assert distillation_cfg.ibot.separate_head is True
        assert distillation_cfg.ibot.head_n_prototypes == self.cfg.ibot.head_n_prototypes
        assert distillation_cfg.dino.head_n_prototypes == self.cfg.dino.head_n_prototypes
        assert distillation_cfg.student.patch_size == self.cfg.student.patch_size

        teacher_model_dict = dict()

        backbone, embed_dim = build_model_from_cfg(distillation_cfg, only_teacher=True)
        teacher_model_dict["backbone"] = backbone

        teacher_model_dict["dino_head"] = DINOHead(
            in_dim=embed_dim,
            out_dim=distillation_cfg.dino.head_n_prototypes,
            hidden_dim=distillation_cfg.dino.head_hidden_dim,
            bottleneck_dim=distillation_cfg.dino.head_bottleneck_dim,
            nlayers=distillation_cfg.dino.head_nlayers,
        )
        teacher_model_dict["ibot_head"] = DINOHead(
            in_dim=embed_dim,
            out_dim=distillation_cfg.ibot.head_n_prototypes,
            hidden_dim=distillation_cfg.ibot.head_hidden_dim,
            bottleneck_dim=distillation_cfg.ibot.head_bottleneck_dim,
            nlayers=distillation_cfg.ibot.head_nlayers,
        )
        self.teacher = nn.ModuleDict(teacher_model_dict)

    def init_weights(self) -> None:
        # All weights are set to `nan` to ensure we initialize everything explicitly
        self.student.backbone.init_weights()
        self.student.dino_head.init_weights()
        self.student.ibot_head.init_weights()
        self.dino_loss.init_weights()
        self.ibot_patch_loss.init_weights()
        self.model_ema.load_state_dict(self.student.state_dict())
        if self.has_gram_teacher:
            if self.gram_ckpt is not None:
                logger.info(f"Loading pretrained weights from {self.gram_ckpt}")
                init_fsdp_model_from_checkpoint(
                    self.gram_teacher,
                    self.gram_ckpt,
                    skip_load_prefixes=[
                        "dino_head",
                        "ibot_head",
                        "dino_loss.center",
                        "ibot_patch_loss.center",
                    ],
                    prefixes_not_sharded=["backbone.rope_embed.periods"],
                    process_group=distributed.get_default_process_group(),
                )
                self.gram_teacher_initialized = True
            else:
                raise ValueError(f"Provide a correct path to {self.gram_ckpt}")
            self.gram_teacher.requires_grad_(False)
            self.gram_teacher.eval()
        if self.cfg.student.resume_from_teacher_chkpt:
            logger.info(f"Loading pretrained weights from {self.cfg.student.resume_from_teacher_chkpt}")
            init_fsdp_model_from_checkpoint(
                self.student,
                self.cfg.student.resume_from_teacher_chkpt,
                skip_load_prefixes=["dino_loss.center", "ibot_patch_loss.center"],
                prefixes_not_sharded=["backbone.rope_embed.periods"],
                process_group=distributed.get_process_subgroup(),
            )
            self.model_ema.load_state_dict(self.student.state_dict())
        if self.cfg.distillation.enabled:
            if self.cfg.distillation.checkpoint_path != "ignore":
                logger.info(f"Loading teacher to distil from : {self.cfg.distillation.checkpoint_path}")
                init_fsdp_model_from_checkpoint(
                    self.teacher,
                    self.cfg.distillation.checkpoint_path,
                    skip_load_prefixes=[],
                )
            else:
                logger.info("Init teacher to distil from, used for testing purpose only")
                self.teacher.backbone.init_weights()
                self.teacher.dino_head.init_weights()
                self.teacher.ibot_head.init_weights()
            logger.info(f"Performing distillation from: {self.teacher}")

    def forward_backward(
        self, data, *, teacher_temp, iteration=0, **ignored_kwargs
    ) -> tuple[Tensor, dict[str, float | Tensor], dict[str, float | Tensor]]:
        del ignored_kwargs
        metrics_dict = {}

        # Shapes
        n_global_crops = 2
        n_local_crops = self.n_local_crops  # self.cfg.crops.local_crops_number
        B = data["collated_local_crops"].shape[0] // n_local_crops
        assert data["collated_global_crops"].shape[0] == n_global_crops * B
        metrics_dict["local_batch_size"] = B
        metrics_dict["global_batch_size"] = data["global_batch_size"]

        global_crops = data["collated_global_crops"].cuda(non_blocking=True)
        local_crops = data["collated_local_crops"].cuda(non_blocking=True)
        masks = data["collated_masks"].cuda(non_blocking=True)
        mask_indices_list = data["mask_indices_list"].cuda(non_blocking=True)
        masks_weight = data["masks_weight"].cuda(non_blocking=True)
        n_masked_patches_tensor = data["n_masked_patches"].cuda(non_blocking=True)

        if self.has_gram_teacher:
            assert "collated_gram_teacher_crops" in data, (
                "no gram teacher crops in the data, have you set cfg.crops.gram_teacher_crops_size?"
            )
            gram_teacher_crops = data["collated_gram_teacher_crops"].cuda(non_blocking=True)
        else:
            gram_teacher_crops = None

        # Teacher output (will trigger an all-gather to unshard)
        teacher_global = self.get_teacher_output(
            global_crops.unflatten(0, (n_global_crops, B)),
            teacher_temp=teacher_temp,
            n_masked_patches_tensor=n_masked_patches_tensor,
            mask_indices_list=mask_indices_list,
            upperbound=data["upperbound"],
        )

        # Student output (will trigger an all-gather to unshard)
        student_global, student_local = self.get_student_output(
            global_crops=global_crops.unflatten(0, (n_global_crops, B)),
            local_crops=local_crops.unflatten(0, (n_local_crops, B)),
            upperbound=data["upperbound"],
            masks=masks,
            mask_indices_list=mask_indices_list,
        )

        # Adding teacher and student for segmentation crops
        seg_extra_loss = 0.0
        if getattr(self.cfg, "seg", None) and self.cfg.seg.enabled and "collated_seg_global_crops" in data:
            n_global_crops = 2
            B = data["collated_global_crops"].shape[0] // n_global_crops

            seg_global = data["collated_seg_global_crops"].cuda(non_blocking=True)
            seg_local  = data["collated_seg_local_crops"].cuda(non_blocking=True)

            # Teacher on segmentation crops
            seg_teacher_global = self.get_teacher_output(
                seg_global.unflatten(0, (n_global_crops, B)),
                teacher_temp=teacher_temp,
                n_masked_patches_tensor=n_masked_patches_tensor,
                mask_indices_list=mask_indices_list,
                upperbound=data["upperbound"],
            )

            # Student on segmentation crops
            seg_student_global, seg_student_local = self.get_student_output(
                global_crops=seg_global.unflatten(0, (n_global_crops, B)),
                local_crops=seg_local.unflatten(0, (self.n_local_crops, B)),
                upperbound=data["upperbound"],
                masks=masks,
                mask_indices_list=mask_indices_list,
            )
            # scales (reuse the ones already computed below or rederive quickly)
            dino_global_terms = n_global_crops * (n_global_crops - 1) if self.dino_global_ignore_diagonal else n_global_crops**2
            dino_local_terms  = n_global_crops * self.n_local_crops
            dino_global_scale = dino_global_terms / (dino_global_terms + dino_local_terms)
            dino_local_scale  = dino_local_terms  / (dino_global_terms + dino_local_terms)

            # (1) Seg -> Seg (student seg vs teacher seg)
            seg2seg_local  = self.dino_loss(seg_student_local["cls_after_head"], seg_teacher_global["cls_centered"])
            seg2seg_global = self.dino_loss(seg_student_global["cls_after_head"], seg_teacher_global["cls_centered"],
                                            ignore_diagonal=self.dino_global_ignore_diagonal)
            seg2seg = dino_local_scale * seg2seg_local + dino_global_scale * seg2seg_global

            # (2) Seg -> Image (student image vs teacher seg)
            seg2img_local  = self.dino_loss(student_local["cls_after_head"], seg_teacher_global["cls_centered"])
            seg2img_global = self.dino_loss(student_global["cls_after_head"], seg_teacher_global["cls_centered"],
                                            ignore_diagonal=self.dino_global_ignore_diagonal)
            seg2img = dino_local_scale * seg2img_local + dino_global_scale * seg2img_global

            # accumulate (weighted)
            # w = float(self.cfg.seg.loss_weight)
            # w = min(0.1, (iteration / 125000) * 0.1)

            
            w = 0.05
            seg_extra_loss = w * (seg2seg + seg2img)

            # # log for visibility
            # loss_dict["seg/seg2seg_local"]  = seg2seg_local
            # loss_dict["seg/seg2seg_global"] = seg2seg_global
            # loss_dict["seg/seg2img_local"]  = seg2img_local
            # loss_dict["seg/seg2img_global"] = seg2img_global
            # loss_dict["seg/weighted_extra"] = seg_extra_loss   
      
        # Gram output
        if self.gram_use_loss:
            gram_global = self.get_gram_teacher_output(
                gram_teacher_crops.unflatten(0, (n_global_crops, B)) if gram_teacher_crops is not None else None,
                masks=masks,
                teacher_global=teacher_global,
                student_global=student_global,
                student_global_crops_size=global_crops.shape[-1],
            )
        else:
            gram_global = {}

        # Clustering
        # Cluster based on local and global crops? Compare student and teacher clustering to each other?
        images = data["collated_images"].cuda(non_blocking=True)
        if self.use_triplet_loss:
            try:
                # TODO: What hyperparameters
                use_teacher = True if self.cfg.triplet.get("cluster_backbone", "student") == "teacher" else False
                # all_emb, all_labels, local_indices = self.get_clustering(images=images,iteration=iteration,use_teacher=use_teacher,eps=self.cfg.triplet.get("clustering_eps", 0.6),min_samples=self.cfg.triplet.get("clustering_min_samples", 4))
                # clustering_gloabal = self.get_hierachical_clustering(images=images,iteration=iteration,use_teacher=use_teacher,min_samples=5,metric="euclidean")
                clustering_gloabal = self.get_hdscan_clustering(
                    images=images,
                    iteration=iteration,
                    use_teacher=False, #hardcoded
                    min_cluster_size=self.cfg.triplet.get("clustering_min_samples", 2),
                    min_samples=2,
                    metric="cosine", #euclidean
                )
            except Exception as err:
                logger.exception("Clustering invocation failed at iteration %s", iteration)
                logger.exception(f"Unexpected {err=}, {type(err)=}")
                clustering_gloabal = {}
        else:
            clustering_gloabal = {}

        # Compute losses and backprop
        loss_accumulator, loss_dict = self.compute_losses(
            teacher_global=teacher_global,
            student_global=student_global,
            student_local=student_local,
            gram_global=gram_global,
            masks=masks,
            mask_indices_list=mask_indices_list,
            masks_weight=masks_weight,
            iteration=iteration,
            clustering_gloabal=clustering_gloabal,
        )

        if getattr(self.cfg, "seg", None) and self.cfg.seg.enabled and "collated_seg_global_crops" in data:
                        # log for visibility
            loss_dict["seg2seg_loss"] = seg2seg * 0.1
            loss_dict["seg2img_loss"] = seg2img * 0.1
            loss_dict["seg/seg2seg_local"]  = seg2seg_local
            loss_dict["seg/seg2seg_global"] = seg2seg_global
            loss_dict["seg/seg2img_local"]  = seg2img_local
            loss_dict["seg/seg2img_global"] = seg2img_global
            loss_dict["seg/weighted_extra"] = seg_extra_loss 
            
        loss_accumulator = loss_accumulator + seg_extra_loss
        
        self.backprop_loss(loss_accumulator)

        # Return total weighted loss, a dict of metrics to log and the los dict
        return loss_accumulator, metrics_dict, loss_dict

    @torch.no_grad()
    def get_teacher_output(
        self,
        images,
        *,
        upperbound,
        mask_indices_list,
        teacher_temp,
        n_masked_patches_tensor,
    ):
        n_crops, B, rgb, H, W = images.shape
        images = images.flatten(0, 1)

        backbone_out = self.teacher.backbone(images, is_training=True)
        cls = backbone_out["x_norm_clstoken"]  # [n_crops * B, D]
        reg = backbone_out["x_storage_tokens"]  # [n_crops * B, R, D]
        ibot_patch = backbone_out["x_norm_patchtokens"]  # [n_crops * B, P, D]

        # IBOT head only on patches that are masked for the student
        buffer = torch.index_select(ibot_patch.flatten(0, 1), dim=0, index=mask_indices_list)
        masked_patch_after_head = self.teacher.ibot_head(buffer)

        # DINO head on CLS tokens
        cls_after_head = self.teacher.dino_head(cls)  # [n_crops * B, K]

        # Center with sinkhorn-knopp
        cls_centered = self.dino_loss.sinkhorn_knopp_teacher(
            cls_after_head, teacher_temp=teacher_temp
        )  # [n_crops * B, K]
        cls_centered = cls_centered.unflatten(0, (n_crops, B))  # [n_crops, B, K]
        masked_patch_centered = self.ibot_patch_loss.sinkhorn_knopp_teacher(
            masked_patch_after_head,
            teacher_temp=teacher_temp,
            n_masked_patches_tensor=n_masked_patches_tensor,
        )  # [n_masked_patches, K]

        return {
            "cls_pre_head": cls.unflatten(0, [n_crops, B]),  # [n_crops, B, D]
            "reg_pre_head": reg.unflatten(0, [n_crops, B]),  # [n_crops, B, R, D]
            "patch_pre_head": ibot_patch.unflatten(0, [n_crops, B]),  # [n_crops, B, P, D]
            "cls_after_head": cls_after_head.unflatten(0, [n_crops, B]),  # [n_crops, B, K]
            "cls_centered": cls_centered,  # [n_crops, B, K]
            "masked_patch_centered": masked_patch_centered,  # [n_masked_patches, K]
        }

    def get_gram_teacher_output(self, images, *, masks, teacher_global, student_global, student_global_crops_size):
        # Get student patch features
        student_patches = student_global["patch_pre_head"].flatten(0, 1)  # [n_crops * B, P, D]

        # Get gram targets
        if self.gram_ema_teacher:
            teacher_patches = teacher_global["patch_pre_head"].flatten(0, 1)  # [n_crops * B, P, D]
        else:
            if not self.gram_teacher_initialized:
                raise ValueError("Gram teacher has not been initialized. Load a checkpoint or from the EMA teacher.")
            n_crops, B, rgb, H, W = images.shape
            images = images.flatten(0, 1)  # [n_crops * B, rgb, H, W]

            with torch.no_grad():
                backbone_out = self.gram_teacher.backbone(images, is_training=True)
            teacher_patches = backbone_out.x_norm_patchtokens  # [n_crops * B, P_T, D]

            # Downsample Gram teacher features if needed
            if teacher_patches.shape[1] != student_patches.shape[1]:
                N = H // self.cfg.student.patch_size
                assert teacher_patches.shape[1] == N**2
                N_student = student_global_crops_size // self.cfg.student.patch_size
                assert student_patches.shape[1] == N_student**2
                patches_hw = teacher_patches.transpose(-2, -1).unflatten(-1, (N, N))  # [n_crops * B, D, N, N]
                patches_hw = torch.nn.functional.interpolate(
                    patches_hw,
                    size=(N_student, N_student),
                    mode=self.gram_global_teacher_resize_method,
                    align_corners=False,
                    antialias=self.gram_global_teacher_resize_antialias,
                )
                teacher_patches = patches_hw.flatten(-2, -1).transpose(
                    -2, -1
                )  # [n_crops * B, N_student * N_student, D]
                assert teacher_patches.shape == student_patches.shape

        # Select the patches to be considered in the loss
        orig_student_patches = student_patches
        orig_teacher_patches = teacher_patches
        if self.gram_tokens_used == "masked":
            student_patches = student_patches[masks]
            teacher_patches = teacher_patches[masks]
        elif self.gram_tokens_used == "unmasked":
            student_patches = student_patches[~masks]
            teacher_patches = teacher_patches[~masks]

        return {
            "student_patches": student_patches,  # [n_crops * B, P, D] or [n_selected_patches, D]
            "teacher_patches": teacher_patches,  # [n_crops * B, P, D] or [n_selected_patches, D]
            # Unmasked patches, for computing statistics
            "orig_student_patches": orig_student_patches,  # [n_crops * B, P, D]
            "orig_teacher_patches": orig_teacher_patches,  # [n_crops * B, P, D]
        }
    
    @torch.no_grad()
    def get_clustering(
        self,
        *,
        images: torch.Tensor | None,
        iteration: int,
        use_teacher: bool = False,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = "euclidean",
    ):
        """
        Compute embeddings for the provided *whole* images, gather across ranks and run DBSCAN.
        Returns:
            all_emb: Tensor (N_total, D) -- gathered embeddings (on device)
            labels_tensor: LongTensor (N_total,) -- cluster labels (-1 => noise)
            local_indices: LongTensor (n_local, ) -- indices inside all_emb that correspond to this rank's samples
        """
        # TODO:
        # - Use student or teacher? Whole image embeddings? What should be eps?
        if images is None:
            return None, None, None

        # select model container (teacher recommended for stability)
        model_container = self.model_ema if use_teacher else self.student
        if "backbone" not in model_container:
            logger.warning("No backbone found in selected model container for clustering.")
            return None, None, None
        backbone = model_container["backbone"]

        # device for backbone
        device = next(backbone.parameters()).device if any(True for _ in backbone.parameters()) else images.device
        imgs = images.to(device, non_blocking=True)

        # get per-rank embeddings
        try:
            backbone_out = backbone(imgs, is_training=False)
        except Exception as e:
            logger.exception("Backbone forward failed during clustering: %s", e)
            return None, None, None

        # support backbone returning either a tensor (B, D) or a dict/list (we handled before)
        if isinstance(backbone_out, torch.Tensor):
            emb_local = backbone_out  # [B_local, D]
        elif isinstance(backbone_out, dict) and "x_norm_clstoken" in backbone_out:
            emb_local = backbone_out["x_norm_clstoken"]
        elif isinstance(backbone_out, (list, tuple)) and len(backbone_out) > 0 and isinstance(backbone_out[0], dict):
            emb_local = backbone_out[0]["x_norm_clstoken"]
        else:
            logger.exception("Unexpected backbone return structure for clustering.")
            return None, None, None

        emb_local = emb_local.detach().contiguous().to(torch.float32)  # [B_local, D]
        emb_local = torch.nn.functional.normalize(emb_local, p=2, dim=1)

        # Gather per-rank embeddings: this returns a list of tensors [emb_rank0, emb_rank1, ...]
        try:
            gathered = distributed.gather_all_tensors(emb_local, group=None)  # list
        except Exception:
            # fallback to CPU gather then cat
            try:
                gathered = [g.cpu() for g in distributed.gather_all_tensors(emb_local.cpu(), group=None)]
            except Exception as e:
                logger.exception("Failed to gather embeddings for clustering: %s", e)
                return None, None, None

        # Build all_emb on this rank
        all_emb = torch.cat(gathered, dim=0)  # (N_total, D)
        N_total = all_emb.shape[0]

        # compute where our local rows are in the concatenation
        rank = distributed.get_rank()
        # compute offset by summing sizes of earlier ranks
        offset = 0
        for r in range(rank):
            offset += gathered[r].shape[0]
        local_len = emb_local.shape[0]
        local_indices = torch.arange(offset, offset + local_len, dtype=torch.long, device=all_emb.device)

        # Run DBSCAN on main process
        labels_tensor = None
        if distributed.is_main_process():
            try:
                all_emb_np = all_emb.cpu().numpy()
                from sklearn.cluster import DBSCAN
                from sklearn.metrics import pairwise_distances

                dists = pairwise_distances(all_emb_np, metric=metric)
                # Take only upper triangle (to avoid double counting and zeros)
                triu = dists[np.triu_indices_from(dists, k=1)]
                mean_dist = triu.mean()
                std_dist = triu.std()
                eps = mean_dist - std_dist
                if eps < 0:
                    eps = mean_dist
                db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, n_jobs=-1)
                labels = db.fit_predict(all_emb_np)  # numpy array length N_total: ints
                # convert to tensor on device and broadcast to all ranks
                try:
                    labels_tensor = torch.from_numpy(labels).to(all_emb.device, dtype=torch.int64)
                    # broadcast (works if backend supports tensors on this device)
                    torch.distributed.broadcast(labels_tensor, src=0)
                except Exception:
                    # fallback: broadcast CPU tensor and move to device
                    cpu_labels = torch.from_numpy(labels).to(torch.int64)
                    torch.distributed.broadcast(cpu_labels, src=0)
                    labels_tensor = cpu_labels.to(all_emb.device)
                # (optional) save to disk for debugging; but you said no disk reliance
                out_dir = Path(self.cfg.train.output_dir) / "clustering" / f"iter_{iteration}"
                out_dir.mkdir(parents=True, exist_ok=True)
                np.save(out_dir / "embeddings.npy", all_emb.cpu().numpy())
                np.save(out_dir / "dbscan_labels.npy", labels.astype(np.int32))
                meta = {
                    "n_points": int(N_total),
                    "n_clusters": int(len(set(labels)) - (1 if -1 in labels else 0)),
                    "n_noise": int((labels == -1).sum()),
                    "eps": float(eps),
                    "min_samples": int(min_samples),
                    "use_teacher": bool(use_teacher),
                }
                np.save(out_dir / "meta.npy", meta)
                logger.info(f"[Clustering] iter={iteration}: pts={meta['n_points']} clusters={meta['n_clusters']} noise={meta['n_noise']} -> {out_dir}")
            except Exception as e:
                logger.exception("DBSCAN failed in get_student_clustering at iteration %s: %s", iteration, e)
                labels_tensor = None
        else:
            # non-main ranks must create a placeholder and receive broadcast
            # Create a placeholder same shape and dtype as expected
            # We'll allocate on the same device as all_emb (then the broadcast above works), with zeros
            labels_tensor = torch.empty((N_total,), dtype=torch.int64, device=all_emb.device)
            try:
                torch.distributed.broadcast(labels_tensor, src=0)
            except Exception:
                # fallback to CPU broadcast: make CPU placeholder, then receive, then move to device
                cpu_labels = torch.empty((N_total,), dtype=torch.int64)
                torch.distributed.broadcast(cpu_labels, src=0)
                labels_tensor = cpu_labels.to(all_emb.device)

        return all_emb, labels_tensor, local_indices
    
    # TODO: EPS schdeule, ignore noise points
    @torch.no_grad()
    def get_hierachical_clustering(
        self,
        *,
        images: torch.Tensor | None,
        iteration: int,
        use_teacher: bool = False,
        min_samples: int = 5,
        metric: str = "euclidean",
    ):
        """
        Compute embeddings for the provided *whole* images, gather across ranks and run DBSCAN at
        three eps levels: [mean - std, mean, mean + std].
        Returns:
            all_emb: Tensor (N_total, D) -- gathered embeddings (on device)
            labels_per_eps: LongTensor (L, N_total) -- DBSCAN labels per eps (L=3)
            centroids_per_eps: list of L tensors, each (n_clusters_j, D) on same device as all_emb
            centroid_labels_per_eps: list of L 1D tensors with the corresponding cluster labels
            local_indices: LongTensor (n_local,) -- indices inside all_emb for this rank's samples
        """
        if images is None:
            return {}

        # select model container (teacher recommended for stability)
        model_container = self.model_ema if use_teacher else self.student
        if "backbone" not in model_container:
            logger.warning("No backbone found in selected model container for clustering.")
            return {}
        backbone = model_container["backbone"]

        # device for backbone
        device = next(backbone.parameters()).device if any(True for _ in backbone.parameters()) else images.device
        imgs = images.to(device, non_blocking=True)

        # get per-rank embeddings
        try:
            backbone_out = backbone(imgs, is_training=False)
        except Exception as e:
            logger.exception("Backbone forward failed during clustering: %s", e)
            return {}

        # support backbone returning either a tensor (B, D) or a dict/list
        if isinstance(backbone_out, torch.Tensor):
            emb_local = backbone_out  # [B_local, D]
        elif isinstance(backbone_out, dict) and "x_norm_clstoken" in backbone_out:
            emb_local = backbone_out["x_norm_clstoken"]
        elif isinstance(backbone_out, (list, tuple)) and len(backbone_out) > 0 and isinstance(backbone_out[0], dict):
            emb_local = backbone_out[0]["x_norm_clstoken"]
        else:
            logger.exception("Unexpected backbone return structure for clustering.")
            return {}

        emb_local = emb_local.detach().contiguous().to(torch.float32)  # [B_local, D]
        emb_local = torch.nn.functional.normalize(emb_local, p=2, dim=1)

        # Gather per-rank embeddings: list of tensors [emb_rank0, emb_rank1, ...]; each rank obtains this list
        try:
            gathered = distributed.gather_all_tensors(emb_local, group=None)  # list
        except Exception:
            # fallback to CPU gather then cat
            try:
                gathered = [g.cpu() for g in distributed.gather_all_tensors(emb_local.cpu(), group=None)]
            except Exception as e:
                logger.exception("Failed to gather embeddings for clustering: %s", e)
                return {}

        # Build all_emb on this rank
        all_emb = torch.cat(gathered, dim=0)  # (N_total, D)
        N_total = all_emb.shape[0]

        # compute where our local rows are in the concatenation
        rank = distributed.get_rank()
        offset = 0
        for r in range(rank):
            offset += gathered[r].shape[0]
        local_len = emb_local.shape[0]
        local_indices = torch.arange(offset, offset + local_len, dtype=torch.long, device=all_emb.device)

        # On main process compute pairwise distances statistics and run DBSCAN for three eps
        L = 1  # three eps levels: mean-std, mean, mean+std
        labels_list = [None] * L
        if distributed.is_main_process():
            try:
                from sklearn.cluster import DBSCAN
                from sklearn.metrics import pairwise_distances
                from sklearn.neighbors import NearestNeighbors
                all_emb_np = all_emb.cpu().numpy()
                dists = pairwise_distances(all_emb_np, metric=metric)
                # upper triangle excluding diagonal
                triu = dists[np.triu_indices_from(dists, k=1)]
                mean_dist = float(triu.mean()) if triu.size > 0 else float(dists.mean())
                std_dist = float(triu.std()) if triu.size > 0 else float(dists.std())

                # --- compute eps via k-distance "elbow" (k = min_samples - 1) ---
                # Choose k for k-distance. For DBSCAN you often use k = min_samples - 1
                k_for_kdistance = max(1, int(min_samples) - 1)

                # Compute distances to k-th nearest neighbor for every point
                # NearestNeighbors returns distances including self at index 0 (distance 0.0)
                nn = NearestNeighbors(n_neighbors=k_for_kdistance + 1, metric=metric, n_jobs=-1)
                nn.fit(all_emb_np)
                distances_all, _ = nn.kneighbors(all_emb_np, return_distance=True)
                # distances_all[:, k_for_kdistance] is the distance to the k-th neighbor (0-based indexing includes self)
                k_distances = distances_all[:, k_for_kdistance]

                # Sort distances ascending as in the k-distance plot
                k_dist_sorted = np.sort(k_distances)

                # If too few points, fallback to mean/std based eps
                def knee_from_sorted(y):
                    n = len(y)
                    if n < 3:
                        return float(np.median(y))
                    # construct x = 0..n-1
                    x = np.arange(n).astype(float)

                    # normalize to [0,1] to make distances scale-invariant
                    x_norm = (x - x[0]) / (x[-1] - x[0]) if x[-1] != x[0] else x
                    y_norm = (y - y[0]) / (y[-1] - y[0]) if y[-1] != y[0] else y

                    # line vector from first to last
                    line_vec = np.array([x_norm[-1] - x_norm[0], y_norm[-1] - y_norm[0]])
                    if np.allclose(line_vec, 0):
                        return float(y[int(n // 2)])
                    # point vectors from first point
                    pts = np.stack([x_norm - x_norm[0], y_norm - y_norm[0]], axis=1)
                    # perpendicular distance from each point to the line
                    # cross product magnitude divided by line length
                    num = np.abs(pts[:, 0] * line_vec[1] - pts[:, 1] * line_vec[0])
                    denom = np.linalg.norm(line_vec)
                    perp_dists = num / (denom + 1e-12)
                    knee_idx = int(np.argmax(perp_dists))
                    return float(y[knee_idx])

                try:
                    eps_knee = knee_from_sorted(k_dist_sorted)
                    # guard against degenerate numeric values
                    if not np.isfinite(eps_knee) or eps_knee <= 0:
                        raise RuntimeError("Invalid eps from knee detection")
                except Exception:
                    # fallback to heuristic: mean + 0.5*std of pairwise distances
                    eps_knee = max(1e-6, mean_dist + 0.5 * std_dist)

                # You asked for 3 eps levels around the knee earlier in the code comment.
                # Use knee and a +/- relative neighborhood as additional robustness:
                eps_vals = [eps_knee, eps_knee * 0.8, eps_knee * 1.25][:L]

                # If L==1 the list above will be truncated to [eps_knee]
                # --- run DBSCAN for each eps ---
                for i, eps_i in enumerate(eps_vals):
                    db = DBSCAN(eps=eps_i, min_samples=min_samples, metric=metric, n_jobs=-1)
                    labels = db.fit_predict(all_emb_np)  # numpy array length N_total: ints
                    labels_list[i] = labels.astype(np.int64)
                # optional debug saving
                meta = {
                    "n_points": int(N_total),
                    "eps_vals": eps_vals,
                    "min_samples": int(min_samples),
                    "use_teacher": bool(use_teacher),
                }
                if self.cfg.triplet.save_clustering:
                    out_dir = Path(self.cfg.train.output_dir) / "clustering" / f"iter_{iteration}"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    np.save(out_dir / "embeddings.npy", all_emb_np)
                    for i, lab in enumerate(labels_list):
                        np.save(out_dir / f"dbscan_labels_eps{i}.npy", lab.astype(np.int32))
                    np.save(out_dir / "meta.npy", meta)
                    logger.info(f"[Clustering] iter={iteration}: pts={meta['n_points']} eps={eps_vals} -> {out_dir}")
            except Exception as e:
                logger.exception("DBSCAN failed in get_clustering at iteration %s: %s", iteration, e)
                labels_list = [np.full((N_total,), -1, dtype=np.int64) for _ in range(L)]
        # Broadcast each labels_list[i] from main to every rank. Create torch tensors for broadcast.
        labels_per_eps = torch.empty((L, N_total), dtype=torch.int64, device=all_emb.device)
        for i in range(L):
            if distributed.is_main_process():
                lab_np = labels_list[i]
                lab_tensor = torch.from_numpy(lab_np).to(all_emb.device, dtype=torch.int64)
            else:
                lab_tensor = torch.empty((N_total,), dtype=torch.int64, device=all_emb.device)
            # broadcast (works for GPU tensors when using NCCL) from rank 0
            try:
                torch.distributed.broadcast(lab_tensor, src=0)
            except Exception:
                # fallback to CPU broadcast if needed
                cpu_tensor = lab_tensor.cpu()
                torch.distributed.broadcast(cpu_tensor, src=0)
                lab_tensor = cpu_tensor.to(all_emb.device)
            labels_per_eps[i] = lab_tensor

        # Now compute centroids on every rank locally from all_emb and labels_per_eps
        centroids_per_eps = []
        centroid_labels_per_eps = []
        for i in range(L):
            lab_tensor = labels_per_eps[i]  # (N_total,)
            # ignore noise
            mask = lab_tensor != -1
            if mask.sum().item() == 0:
                centroids_per_eps.append(torch.empty((0, all_emb.shape[1]), dtype=all_emb.dtype, device=all_emb.device))
                centroid_labels_per_eps.append(torch.empty((0,), dtype=torch.int64, device=all_emb.device))
                continue
            labels_i = lab_tensor.cpu().numpy()
            unique_labels = np.unique(labels_i[labels_i != -1])
            centroids = []
            cent_labels = []
            for lab in unique_labels:
                # boolean mask (on CPU or device)
                idxs = np.nonzero(labels_i == int(lab))[0]
                if idxs.size == 0:
                    continue
                # compute centroid in torch for numerical stability
                rows = all_emb[idxs]  # (n_i, D)
                centroid = rows.mean(dim=0)
                centroids.append(centroid)
                cent_labels.append(int(lab))
            if len(centroids) == 0:
                centroids_tensor = torch.empty((0, all_emb.shape[1]), dtype=all_emb.dtype, device=all_emb.device)
                cent_label_tensor = torch.empty((0,), dtype=torch.int64, device=all_emb.device)
            else:
                centroids_tensor = torch.stack(centroids, dim=0)  # (n_clusters_i, D)
                cent_label_tensor = torch.tensor(cent_labels, dtype=torch.int64, device=all_emb.device)
            # normalize centroids (cosine)
            if centroids_tensor.numel() > 0:
                centroids_tensor = torch.nn.functional.normalize(centroids_tensor, p=2, dim=1)
            centroids_per_eps.append(centroids_tensor)
            centroid_labels_per_eps.append(cent_label_tensor)
        
        return {
            "embed": all_emb,
            "labels_per_eps": labels_per_eps,
            "centroids_per_eps": centroids_per_eps,
            "centroid_labels_per_eps": centroid_labels_per_eps,
            "local_indices": local_indices,
        }
    
    def get_hdscan_clustering(
        self,
        *,
        images: torch.Tensor | None,
        iteration: int,
        use_teacher: bool = False,
        min_cluster_size: int = 2,
        min_samples: int = 1,
        metric: str = "euclidean",
    ):
        """
        Compute embeddings for the provided images, gather across ranks, run HDBSCAN,
        and build hierarchy-driven triplet data.

        Returns:
            all_emb: Tensor (N_total, D) -- gathered embeddings (on device)
            positives: list[list[np.ndarray]] -- parent centroids along each point's path (depth>0)
            negatives: list[list[np.ndarray]] -- negatives per our hierarchy rules
            lambdas:   list[list[float]] -- lambda_leave values aligned with positives
            paths:     dict[int -> list[dict]] -- trimmed paths with QA info (negatives_used on final step)
            local_indices: LongTensor (n_local,) -- indices into all_emb for this rank's samples
        """
        if images is None:
            return {
                    "embed": None,
                    "positives": None,
                    "negatives": None,
                    "lambdas": None,
                    "paths": None,
                    "local_indices": None,
                }

        model_container = self.student #self.model_ema if use_teacher else self.student
        if "backbone" not in model_container:
            logger.warning("No backbone found in selected model container for clustering.")
            return {
                    "embed": None,
                    "positives": None,
                    "negatives": None,
                    "lambdas": None,
                    "paths": None,
                    "local_indices": None,
                }
        backbone = model_container["backbone"]

        # device for backbone
        device = next(backbone.parameters()).device if any(True for _ in backbone.parameters()) else images.device
        imgs = images.to(device, non_blocking=True)

        # get per-rank embeddings
        try:
            backbone_out = backbone(imgs, is_training=False)
        except Exception as e:
            logger.exception("Backbone forward failed during hierarchical clustering: %s", e)
            return {
                    "embed": None,
                    "positives": None,
                    "negatives": None,
                    "lambdas": None,
                    "paths": None,
                    "local_indices": None,
                }

        # support backbone returning either a tensor (B, D) or a dict/list
        if isinstance(backbone_out, torch.Tensor):
            emb_local = backbone_out  # [B_local, D]
        elif isinstance(backbone_out, dict) and "x_norm_clstoken" in backbone_out:
            emb_local = backbone_out["x_norm_clstoken"]
        elif isinstance(backbone_out, (list, tuple)) and len(backbone_out) > 0 and isinstance(backbone_out[0], dict):
            emb_local = backbone_out[0]["x_norm_clstoken"]
        else:
            logger.exception("Unexpected backbone return structure for hierarchical clustering.")
            return {
                    "embed": None,
                    "positives": None,
                    "negatives": None,
                    "lambdas": None,
                    "paths": None,
                    "local_indices": None,
                }

        emb_local = emb_local.detach().contiguous().to(torch.float32)  # [B_local, D]
        emb_local = torch.nn.functional.normalize(emb_local, p=2, dim=1)

        # Gather embeddings across ranks
        try:
            gathered = distributed.gather_all_tensors(emb_local, group=None)
        except Exception:
            try:
                gathered = [g.cpu() for g in distributed.gather_all_tensors(emb_local.cpu(), group=None)]
            except Exception as e:
                logger.exception("Failed to gather embeddings for hierarchical clustering: %s", e)
                return {
                    "embed": None,
                    "positives": None,
                    "negatives": None,
                    "lambdas": None,
                    "paths": None,
                    "local_indices": None,
                }

        all_emb = torch.cat(gathered, dim=0).to(torch.float32)  # (N_total, D)
        N_total = all_emb.shape[0]

        # compute local indices into all_emb
        rank = distributed.get_rank()
        offset = 0
        for r in range(rank):
            offset += gathered[r].shape[0]
        local_len = emb_local.shape[0]
        local_indices = torch.arange(offset, offset + local_len, dtype=torch.long, device=all_emb.device)

        # Everyone can run HDBSCAN locally; all ranks have all_emb, so no broadcast needed
        try:
            all_emb = torch.nn.functional.normalize(all_emb, p=2, dim=1) #THIS DOES COSINE FOR US, HDBSCAN DOES NOT SUPPORT COSINE RN
            all_emb_np = all_emb.cpu().numpy()
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='euclidean', #metric,
                cluster_selection_method='eom',
                prediction_data=False,
            ).fit(all_emb_np)

            # Build triplet lists & QA paths
            positives, negatives, lambdas, neg_lambdas, paths = build_triplet_lists_with_paths(all_emb_np, clusterer)

            # optional debug saving
            meta = {
                "n_points": N_total,
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "metric": metric,
                "use_teacher": use_teacher,
            }
            if getattr(self.cfg.triplet, "save_clustering", False):
                out_dir = Path(self.cfg.train.output_dir) / "clustering_hdbscan" / f"iter_{iteration}"
                out_dir.mkdir(parents=True, exist_ok=True)
                np.save(out_dir / "embeddings.npy", all_emb_np)
                # pickle-like ragged saves (use numpy object arrays)
                np.save(out_dir / "positives.npy", np.array(positives, dtype=object))
                np.save(out_dir / "negatives.npy", np.array(negatives, dtype=object))
                np.save(out_dir / "lambdas.npy",   np.array(lambdas, dtype=object))
                np.save(out_dir / "neg_lambdas.npy",   np.array(neg_lambdas, dtype=object))
                # save a light JSON for paths (without raw vectors)
                import json
                slim_paths = {
                    int(i): [
                        {
                            k: (float(v) if isinstance(v, (np.floating,)) else v)
                            for k, v in step.items()
                            if k not in ('parent_centroid','child_centroid','negatives_used')
                        }
                        for step in steps
                    ]
                    for i, steps in paths.items()
                }
                (out_dir / "paths.json").write_text(json.dumps(slim_paths, indent=2))
                np.save(out_dir / "meta.npy", meta)
                logger.info(f"[HDBSCAN] iter={iteration}: pts={meta['n_points']} mcs={min_cluster_size} ms={min_samples} -> {out_dir}")

        except Exception as e:
            logger.exception("HDBSCAN pipeline failed in get_hierachical_clustering at iteration %s: %s", iteration, e)
            # graceful fallback: empty outputs with correct arity
            positives = [[] for _ in range(N_total)]
            negatives = [[] for _ in range(N_total)]
            lambdas   = [[] for _ in range(N_total)]
            neg_lambdas = [[] for _ in range(N_total)]
            paths     = {i: [] for i in range(N_total)}

        return {
            "embed": all_emb,
            "positives": positives,
            "negatives": negatives,
            "lambdas": lambdas,
            "neg_lambdas": neg_lambdas,
            "paths": paths,
            "local_indices": local_indices,
        }
    
    # triplet_loss_val, triplet_stats = self.triplet_centroid_loss(
    # anchors=student_anchors,
    # centroids_per_eps=clustering_gloabal["centroids_per_eps"],
    # centroid_labels_per_eps=clustering_gloabal["centroid_labels_per_eps"],
    # labels_per_eps=clustering_gloabal["labels_per_eps"],
    # local_indices=clustering_gloabal["local_indices"],
    # centroid_weights_per_eps=clustering_gloabal.get("centroid_weights_per_eps", None),
    # margin=float(self.cfg.triplet.get("margin", self.triplet_centroid_loss.margin)),

    def get_student_output(self, *, global_crops, local_crops, upperbound, masks, mask_indices_list):
        n_global_crops, B, rgb, H, W = global_crops.shape
        n_local_crops, B, rgb, H, W = local_crops.shape

        global_crops = global_crops.flatten(0, 1)

        # Forward global and local crops through the student backbone jointly
        global_out, local_out = self.student.backbone(
            [global_crops, local_crops.flatten(0, 1)],
            masks=[masks if not self.is_distillation_enabled else None, None],
            is_training=True,
        )
        g_cls, g_reg, g_patch = (
            global_out["x_norm_clstoken"],
            global_out["x_storage_tokens"],
            global_out["x_norm_patchtokens"],
        )
        l_cls, l_reg, l_patch = (
            local_out["x_norm_clstoken"],
            local_out["x_storage_tokens"],
            local_out["x_norm_patchtokens"],
        )

        # IBOT head only on masked patches
        masked_patches_pre_head = torch.index_select(g_patch.flatten(0, 1), dim=0, index=mask_indices_list)
        global_masked_patch_after_head = self.student.ibot_head(masked_patches_pre_head)

        # DINO head on CLS tokens (all in one pass)
        buffer = [
            g_cls,  # [n_global_crops * B, D]
            l_cls,  # [n_local_crops * B, D]
        ]
        sizes = [x.shape[0] for x in buffer]
        buffer = torch.cat(buffer, dim=0)  # [n_global_crops * B + n_local_crops * B, D]
        buffer = self.student.dino_head(buffer)  # [n_global_crops * B + n_local_crops * B, K]
        buffer = torch.split_with_sizes(buffer, sizes, dim=0)

        global_out = {
            "cls_pre_head": g_cls.unflatten(0, [n_global_crops, B]),  # [n_global_crops, B, D]
            "reg_pre_head": g_reg.unflatten(0, [n_global_crops, B]),  # [n_global_crops, B, R, D]
            "patch_pre_head": g_patch.unflatten(0, [n_global_crops, B]),  # [n_global_crops, B, P, D]
            "cls_after_head": buffer[0].unflatten(0, [n_global_crops, B]),  # [n_global_crops, B, K],
            "masked_patch_after_head": global_masked_patch_after_head,  # [n_masked_patches, K]
            "masked_patch_pre_head": masked_patches_pre_head,  # [n_masked_patches, D]
        }
        local_out = {
            "cls_pre_head": l_cls.unflatten(0, [n_local_crops, B]),  # [n_local_crops, B, D]
            "reg_pre_head": l_reg.unflatten(0, [n_local_crops, B]),  # [n_local_crops, B, R, D]
            "patch_pre_head": l_patch.unflatten(0, [n_local_crops, B]),  # [n_local_crops, B, P, D]
            "cls_after_head": buffer[1].unflatten(0, [n_local_crops, B]),  # [n_local_crops, B, K],
        }

        return global_out, local_out

    def compute_losses(
        self,
        *,
        teacher_global,
        student_global,
        student_local,
        gram_global,
        masks,
        mask_indices_list,
        masks_weight,
        iteration,
        clustering_gloabal,
    ):
        n_global_crops = student_global["cls_after_head"].shape[0]
        n_local_crops = student_local["cls_after_head"].shape[0]
        loss_dict = {}
        loss_accumulator = 0.0

        # Loss scales like in DINOv2, these are multiplied with the loss weights from the config
        dino_global_terms = (
            n_global_crops * (n_global_crops - 1) if self.dino_global_ignore_diagonal else n_global_crops**2
        )
        dino_local_terms = n_global_crops * n_local_crops
        dino_global_scale = dino_global_terms / (dino_global_terms + dino_local_terms)
        dino_local_scale = dino_local_terms / (dino_global_terms + dino_local_terms)
        koleo_scale = n_global_crops

        # DINO local loss: compare post-head CLS tokens: student(local crops) vs. teacher(global crops)
        dino_local_crops_loss = self.dino_loss(
            student_logits=student_local["cls_after_head"],
            teacher_probs=teacher_global["cls_centered"],
        )
        loss_dict["dino_local_crops_loss"] = dino_local_crops_loss

        # Reweighting of DINO loss
        if self.cfg.dino.reweight_dino_local_loss:
            local_weight = self.dino_local_loss_schedule[iteration]
        else:
            local_weight = 1.0

        loss_dict["dino_local_loss_weight"] = local_weight
        loss_accumulator += self.dino_loss_weight * dino_local_scale * local_weight * dino_local_crops_loss

        # DINO global loss: compare post-head CLS tokens: student(global crops) vs. teacher(global crops)
        dino_global_crops_loss = self.dino_loss(
            student_logits=student_global["cls_after_head"],
            teacher_probs=teacher_global["cls_centered"],
            ignore_diagonal=self.dino_global_ignore_diagonal,
        )
        loss_dict["dino_global_crops_loss"] = dino_global_crops_loss
        loss_accumulator += self.dino_loss_weight * dino_global_scale * dino_global_crops_loss

        # Koleo: regularize pre-head CLS tokens of student(global crops)
        koleo_loss = sum(self.koleo_loss(x) for x in student_global["cls_pre_head"]) / n_global_crops
        loss_dict["koleo_loss"] = koleo_loss
        loss_accumulator += self.dino_koleo_loss_weight * koleo_scale * koleo_loss

        # IBOT loss
        ibot_patch_loss = self.ibot_patch_loss.forward_masked(
            student_global["masked_patch_after_head"],
            teacher_global["masked_patch_centered"],
            student_masks_flat=masks,
            n_masked_patches=mask_indices_list.shape[0],
            masks_weight=masks_weight,
        )
        loss_dict["ibot_loss"] = ibot_patch_loss
        loss_accumulator += self.ibot_loss_weight * ibot_patch_loss

        # Gram loss
        if self.gram_use_loss:
            gram_loss = self.gram_loss(
                gram_global["student_patches"],
                gram_global["teacher_patches"],
                img_level=self.gram_img_level,
            )

            if self.gram_loss_schedule is not None:
                gram_loss_weight = self.gram_loss_schedule[iteration]
            else:
                gram_loss_weight = self.gram_loss_weight

            loss_dict["gram_loss_weight"] = gram_loss_weight
            loss_accumulator += gram_loss * gram_loss_weight
            loss_dict["gram_loss"] = gram_loss

            if self.gram_compute_stats:
                with torch.no_grad():
                    # Save stats over masked / unmasked tokens
                    gram_loss_masked = self.gram_loss(
                        gram_global["orig_student_patches"][masks].detach(),
                        gram_global["orig_teacher_patches"][masks],
                        img_level=False,
                    )
                    loss_dict["stats_only/masked_gram_loss"] = gram_loss_masked
                    gram_loss_unmasked = self.gram_loss(
                        gram_global["orig_student_patches"][~masks].detach(),
                        gram_global["orig_teacher_patches"][~masks],
                        img_level=False,
                    )
                    loss_dict["stats_only/unmasked_gram_loss"] = gram_loss_unmasked

        # inside compute_losses, where triplet_emb, triplet_labels, triplet_local_indices are available
        # triplet loss
        if 1 == 1: #getattr(self.cfg, "triplet", None) and self.cfg.triplet.enabled:
            try:
                # anchors: student embeddings (global crop 0). Keep gradient on student anchors.
                # student_global["cls_pre_head"] shape: [n_global_crops, B_local, D]
                student_anchors = student_global["cls_pre_head"][0]  # [B_local, D]

                # deterministic seed using iteration + rank*const
                rank = distributed.get_rank() if hasattr(distributed, "get_rank") else 0
                # seed = int(iteration) + int(rank) * 10007

                # mode selection from cfg: batch-hard if cfg.triplet.batch_hard True
                # mode = "batch_hard" if self.cfg.triplet.get("batch_hard", False) else "sample"
                triplet_loss_val, triplet_stats = self.triplet_loss(
                    anchors=student_anchors,
                    positives=clustering_gloabal["positives"],
                    negatives=clustering_gloabal["negatives"],
                    lambdas=clustering_gloabal["lambdas"],
                    neg_lambdas=clustering_gloabal.get("neg_lambdas", None),
                    local_indices=clustering_gloabal["local_indices"],
                    margin=float(self.cfg.triplet.get("margin", self.triplet_loss.margin)),
                )
                # triplet_loss_val, triplet_stats = self.triplet_loss(anchors=student_anchors,global_emb=triplet_emb,global_labels=triplet_labels,local_indices=triplet_local_indices,margin=float(self.cfg.triplet.get("margin", self.triplet_loss.margin)),seed=seed,mode=mode,)
                loss_dict["triplet_loss"] = triplet_loss_val
                loss_accumulator += 0.1 * triplet_loss_val #float(self.cfg.triplet.weight) * triplet_loss_val

                # COMPREHENSIVE GRADIENT DEBUGGING
                # Check if anchors require grad
                loss_dict["triplet/anchors_requires_grad"] = float(1.0 if student_anchors.requires_grad else 0.0)
                loss_dict["triplet/tval_requires_grad"] = float(1.0 if getattr(triplet_loss_val, "requires_grad", False) else 0.0)
                loss_dict["triplet/anchors_is_leaf"] = float(1.0 if student_anchors.is_leaf else 0.0)
                loss_dict["triplet/tval_is_leaf"] = float(1.0 if triplet_loss_val.is_leaf else 0.0)
                
                # Check if anchors have grad_fn
                loss_dict["triplet/anchors_grad_fn"] = float(1.0 if student_anchors.grad_fn is not None else 0.0)
                loss_dict["triplet/tval_grad_fn"] = float(1.0 if triplet_loss_val.grad_fn is not None else 0.0)
                
                # Check if student backbone parameters require grad
                backbone_requires_grad = any(p.requires_grad for p in self.student["backbone"].parameters())
                loss_dict["triplet/backbone_requires_grad"] = float(1.0 if backbone_requires_grad else 0.0)
                
                # Check student backbone training mode
                backbone_training = self.student["backbone"].training
                loss_dict["triplet/backbone_training"] = float(1.0 if backbone_training else 0.0)
                
                # Try gradient computation
                try:
                    g = torch.autograd.grad(triplet_loss_val, student_anchors, retain_graph=True, allow_unused=False)[0]
                    loss_dict["triplet/anchor_grad_norm"] = g.norm().detach()
                    loss_dict["triplet/grad_computation_success"] = 1.0
                except Exception as e:
                    loss_dict["triplet/anchor_grad_norm"] = 0.0
                    loss_dict["triplet/grad_computation_success"] = 0.0
                    loss_dict["triplet/grad_error"] = str(type(e).__name__)
                    logger.warning(f"Gradient computation failed: {e}")

                # record stats
                valid_count = int(triplet_stats.get("valid_count", 0))
                total_anchors = int(triplet_stats.get("total_anchors", student_anchors.shape[0]))
                loss_dict["triplet/valid_anchors"] = valid_count
                loss_dict["triplet/total_anchors"] = total_anchors
                loss_dict["triplet/weighting_mode"] = triplet_stats.get("weighting_mode", "None")
                loss_dict["triplet/negative_weighting"] = triplet_stats.get("negative_weighting", "None")
                loss_dict["triplet/lambda_scaling"] = triplet_stats.get("lambda_scaling", "None")
                logger.info(f"[Triplet HDBScan] iter={iteration}: negative_weighting={loss_dict['triplet/negative_weighting']} lambda_scaling={loss_dict['triplet/lambda_scaling']} mode={loss_dict['triplet/weighting_mode']} valid_anchors={valid_count}/{total_anchors}")

            except Exception as err:
                logger.exception("Triplet loss failed at iteration %s", iteration)
                logger.exception(f"Unexpected {err=}, {type(err)=}")
                loss_dict["triplet_loss"] = 0
                loss_dict["triplet/valid_anchors"] = 0
                loss_dict["triplet/total_anchors"] = 0
                loss_dict["triplet/weighting_mode"] = "NaN"
        else:
            loss_dict["triplet_loss"] = 0
            loss_dict["triplet/valid_anchors"] = 0
            loss_dict["triplet/total_anchors"] = 0
            loss_dict["triplet/weighting_mode"] = "NaN"

        return loss_accumulator, loss_dict
    
# Questions to talk about
# - Use use_teacher=True for clustering because EMA/teacher embeddings are stable (less noisy clustering). The EMA teacher changes slowly and therefore produces cluster assignments that vary less across iterations — that helps DBSCAN produce sensible clusters instead of oscillating.
# - Triplet anchors are the student embeddings because we need gradients to update the student. The loss must produce gradients for the network parameters we train. The EMA teacher is typically requires_grad=False (or detached); if you used teacher embeddings as anchors you would not get gradients to update the student (or you'd have to create a new computational graph that forces gradient flow through student in a different way). So the usual pattern is: teacher creates stable clusters/targets, student is trained against them.
# - Perform clustering on the whole images, or local/global crops? -> global image
# - DBSCAN eps selection matters; heuristics (mean distance ± std)?  -> Find eps s.t. only one cluster and then decrease until B//2 clusters
# - If clusters are tiny/rare, triplet signals may be sparse. Small batchsizes make stuff weird
# - If we scale to very many samples (e.g., whole dataset clustering every epoch), then the memory / compute for DBSCAN may become large


    @torch.no_grad()
    def gram_load_ema_teacher(self):
        if self.has_gram_teacher:
            skip_load_prefixes = ["dino_head.", "ibot_head."]
            self.gram_teacher.load_state_dict(
                {
                    k: v
                    for k, v in self.model_ema.state_dict().items()
                    if not any(k.startswith(prefix) for prefix in skip_load_prefixes)
                }
            )
            self.gram_teacher.requires_grad_(False)
            self.gram_teacher.eval()
            self.gram_teacher_initialized = True

    def train(self):
        super().train()
        self.teacher.eval()
        if self.has_gram_teacher:
            self.gram_teacher.eval()

    def forward(self, inputs):
        raise NotImplementedError

    def backprop_loss(self, loss):
        loss.backward()

    def update_ema(self, m):
        if self.ema_params_lists is None:
            student_param_list = []
            teacher_param_list = []
            for k in self.student.keys():
                for ms, mt in zip(self.student[k].parameters(), self.model_ema[k].parameters()):
                    student_param_list += [ms]
                    teacher_param_list += [mt]
            self.ema_params_lists = (student_param_list, teacher_param_list)
        else:
            student_param_list, teacher_param_list = self.ema_params_lists
        with torch.no_grad():
            torch._foreach_mul_(teacher_param_list, m)
            torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - m)

    def update_gram(self, m=0):
        if not self.has_gram_teacher:
            return
        logger.info("Updating gram teacher with teacher weights.")
        if self.gram_params_lists is None:
            teacher_param_list = []
            gramteacher_param_list = []
            for k in self.gram_teacher.keys():
                for mgt, mt in zip(self.gram_teacher[k].parameters(), self.teacher[k].parameters()):
                    gramteacher_param_list += [mgt]
                    teacher_param_list += [mt]
            self.gram_params_lists = (gramteacher_param_list, teacher_param_list)
        else:
            gramteacher_param_list, teacher_param_list = self.gram_params_lists

        with torch.no_grad():
            torch._foreach_mul_(gramteacher_param_list, m)
            torch._foreach_add_(gramteacher_param_list, teacher_param_list, alpha=1 - m)

    def build_data_augmentation_dino(self, cfg):
        return DataAugmentationDINO(
            cfg.crops.global_crops_scale,
            cfg.crops.local_crops_scale,
            cfg.crops.local_crops_number,
            global_crops_size=cfg.crops.global_crops_size,
            local_crops_size=cfg.crops.local_crops_size,
            gram_teacher_crops_size=cfg.crops.gram_teacher_crops_size,
            gram_teacher_no_distortions=cfg.crops.gram_teacher_no_distortions,
            local_crops_subset_of_global_crops=cfg.crops.localcrops_subset_of_globalcrops,
            share_color_jitter=cfg.crops.share_color_jitter,
            horizontal_flips=cfg.crops.horizontal_flips,
            mean=cfg.crops.rgb_mean,
            std=cfg.crops.rgb_std,
        )

    def get_maybe_fused_params_for_submodel(self, m: nn.Module):
        params_groups = get_params_groups_with_decay_fsdp(
            model=m,
            lr_decay_rate=self.cfg.optim.layerwise_decay,
            patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult,
            dino_head_wd_multiplier=self.cfg.optim.dino_head_wd_multiplier,
        )
        if self.cfg.optim.multi_tensor_optim:
            fused_params_groups = fuse_params_groups(params_groups)
            logger.info("fusing param groups")

            for g in fused_params_groups:
                g["foreach"] = True
                g["fused"] = True
            return fused_params_groups
        else:
            return params_groups

    def get_params_groups(self):
        all_params_groups = []
        for name, m in self.student.items():
            logger.info(f"Getting paramer groups for {name}")
            all_params_groups += self.get_maybe_fused_params_for_submodel(m)
        return all_params_groups

    def prepare_for_distributed_training(self) -> None:
        process_subgroup = distributed.get_process_subgroup()
        default_process_group = distributed.get_default_process_group()
        inference_only_models = [self.model_ema]
        inference_only_models_process_groups = [process_subgroup]
        if self.has_gram_teacher:
            inference_only_models.append(self.gram_teacher)
            inference_only_models_process_groups.append(default_process_group)
        if self.cfg.distillation.enabled:
            inference_only_models.append(self.teacher)
            inference_only_models_process_groups.append(default_process_group)
        ac_compile_parallelize(
            trained_model=self.student,
            inference_only_models=inference_only_models,
            cfg=self.cfg,
            trained_model_process_group=process_subgroup,
            inference_only_models_process_groups=inference_only_models_process_groups,
        )

    def broadcast_to_subgroups(self, tensor, over_dim, global_batch_size=None):
        """
        This is an operation that takes a tensor from the default process group, gathers it, stacks it, then scatters it within a smaller process subgroup
        """
        world_size = distributed.get_world_size()
        subgroup_size = distributed.get_subgroup_size()
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]

        torch.distributed.all_gather(gathered, tensor)
        catted = torch.cat(gathered, dim=over_dim)
        if global_batch_size is not None:
            catted = catted.narrow(dim=over_dim, start=0, length=global_batch_size)

        return catted.chunk(subgroup_size, dim=over_dim)[distributed.get_subgroup_rank()].clone()
