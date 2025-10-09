from dinov3.configs import DinoV3SetupArgs, get_cfg_from_args, apply_scaling_rules_to_cfg
import dinov3.distributed as distributed
from dinov3.train import SSLMetaArch
from dinov3.checkpointer import find_latest_checkpoint, load_checkpoint
from pathlib import Path
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Distributed setup required for dinov3
if not distributed.is_enabled():
    distributed.enable(
        overwrite=True,
        nccl_async_error_handling=True,
        restrict_print_to_main_process=True,
        timeout=None,
    )

def load_backbone(
    config_file: str,
    pretrained_weights: str,
    output_dir: str,
):
    args = DinoV3SetupArgs(
        config_file=config_file,
        pretrained_weights=pretrained_weights,
        shard_unsharded_model=False,
        output_dir=output_dir,
    )
    cfg = get_cfg_from_args(args, strict=False)
    apply_scaling_rules_to_cfg(cfg)
    model = SSLMetaArch(cfg)
    print("Materializing model parameters on", DEVICE)
    model = model.to_empty(device=DEVICE) 
    ckpt_dir = Path(cfg.train.output_dir, "ckpt").expanduser()
    last_checkpoint_dir = find_latest_checkpoint(ckpt_dir)
    process_subgroup = distributed.get_process_subgroup()
    start_iter = (
        load_checkpoint(
            last_checkpoint_dir,
            model=model,
            optimizer=None,
            strict_loading=False,
            process_group=process_subgroup,
        )
        + 1
    )
    print(f"Model loaded on {start_iter} start iteration")
    embedding_model = model.student.backbone
    embedding_model.eval()
    print("Model backbone architecture: \n", embedding_model)
    return embedding_model
