# Hierarchy-Aware Self-Supervised Learning for Biological Cell Images

[ :scroll: [`Paper`](#)] [ :book: [`BibTeX`](#citing-this-work)]

Self-supervised vision models applied to biological cell images suffer from a systematic failure mode: coarse imaging factors (e.g., acquisition modality, staining protocol) dominate the learned representation, overwhelming the fine morphological signals that distinguish biologically distinct cell subtypes. The result is a latent space where semantically different cells appear identical and hierarchically related subtypes collapse into the same cluster.

We introduce a **hierarchy-aware self-supervised training framework** built on top of DINOv3 that directly counteracts this tendency via two tightly integrated components:

1. **Double-Teacher Distillation** — A segmentation teacher is incorporated alongside the standard self-supervised teacher. By supervising patch-level features with segmentation priors, the student network learns morphologically aware representations that are sensitive to cell shape and boundary structure rather than imaging modality alone.

2. **HDBSCAN Contrastive Loss** — At each training step, HDBSCAN is run on the current embedding space to discover the latent cluster hierarchy. A contrastive loss is then derived from this hierarchy, penalizing embeddings that violate hierarchical separation at any granularity. This steers the model toward decision boundaries that respect biological subtypes at multiple levels of specificity.

Together, these two components push the embedding space toward a structure that is simultaneously morphologically grounded and hierarchically consistent — enabling meaningful sub-cluster discovery driven by fine morphological detail rather than confounded by acquisition artifacts.

---

## Pretrained Checkpoints

<table style="margin: auto">
  <thead>
    <tr>
      <th>Model</th>
      <th>Description</th>
      <th>Download</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Double Teacher</td>
      <td>Pretrained with segmentation distillation teacher</td>
      <td align="center">[link]</td>
    </tr>
    <tr>
      <td>HDBSCAN Fine-tuned</td>
      <td>Fine-tuned on top of Double Teacher with hierarchy-aware contrastive loss</td>
      <td align="center">[link]</td>
    </tr>
  </tbody>
</table>

---

## Installation

The training and evaluation code requires PyTorch and a [Weights & Biases](https://wandb.ai/) account for experiment tracking. Clone the repository and create the conda environment:

```shell
micromamba env create -f conda.yaml
micromamba activate dinov3
```

---

## Training

### Step 1 — Configure Weights & Biases

Set the following environment variables before launching training. The trainer reads these at runtime — no credentials are stored in the codebase.

| Variable | Description | Required |
|---|---|---|
| `WANDB_ENTITY` | Your W&B username or team name | Yes |
| `WANDB_PROJECT` | Project name for this run | No (defaults to `dinov3-cell`) |
| `WANDB_TAGS` | Comma-separated list of run tags | No |

```shell
export WANDB_ENTITY="<your-wandb-entity>"
export WANDB_PROJECT="<your-project-name>"
export WANDB_TAGS="dinov3,hdbscan,finetuning"   # optional
```

### Step 2 — Prepare the Training Manifest

**Option A — Use our dataset.**
We provide pre-built manifests for our training and evaluation sets. Download the dataset and manifests:

| File | Description | Download |
|------|-------------|----------|
| Cell image dataset | All `.npy` cell images used for training and evaluation | [link] |
| `manifest_train_fixed.csv.gz` | Training manifest (included in repo) | — |
| `manifest_test_fixed.csv.gz` | Evaluation manifest (included in repo) | — |

After downloading the dataset, update the `img_path` and `mask_dir` columns in the manifests to reflect the location of the dataset on your machine:

```python
import gzip, csv, io

OLD_PREFIX = "path_to_dataset/"   # placeholder in the provided manifests
NEW_PREFIX = "/your/local/path/to/dataset/"

for fname in ["manifest_train_fixed.csv.gz", "manifest_test_fixed.csv.gz"]:
    with gzip.open(fname, "rt", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    for row in rows:
        for col in ("img_path", "mask_dir"):
            if col in row:
                row[col] = row[col].replace(OLD_PREFIX, NEW_PREFIX)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    with gzip.open(fname, "wt", newline="") as f:
        f.write(buf.getvalue())
```

**Option B — Bring your own dataset.**
If you are training on a custom cell image collection, generate the manifest using `build_manifest_csv()` provided in `dinov3/data/datasets/n_cells.py`:

```python
from dinov3.data.datasets.n_cells import build_manifest_csv

build_manifest_csv(
    root="/path/to/your/dataset/root",
    split="train",
    out_csv_gz="manifest_train.csv.gz",
)
```

The dataset root must follow this directory structure:

```
<root>/
  <origin>/          # e.g. N_PanNuke, N_MoNuSeg, …
    <split>/         # train / val / test
      [<label>/]     # optional label sub-directories
        original/    # cell images as .npy files  (H × W × 3, uint8 or float32)
        mask/        # binary instance masks as .npy files
```

The manifest is a gzip-compressed CSV with the following schema:

| Column       | Description                                       |
|--------------|---------------------------------------------------|
| `img_path`   | Absolute path to the cell image (`.npy`)          |
| `origin`     | Source dataset name (e.g. `N_PanNuke`)            |
| `label`      | Cell type or class label                          |
| `mask_dir`   | Directory containing the corresponding mask file  |
| `has_empty`  | `1` if an empty-mask variant exists, else `0`     |
| `stem`       | Filename stem (no extension)                      |
| `h`          | Height of the image in pixels                     |
| `w`          | Width of the image in pixels                      |
| `area`       | Area of the image in pixels (`h × w`)             |

### Step 3 — Register Cell Classes

Add your cell origin names from the manifest file to:

```
dinov3/data/datasets/n_cells.py  (line 27)
```

### Step 4 — Launch Training

Run DINOv3 pre-training with the double-teacher distillation and hierarchy-aware contrastive loss on a single node:

```shell
PYTHONPATH=${PWD} python -m dinov3.run.submit dinov3/train/train.py \
  --nodes 1 \
  --config-file dinov3/configs/train/vitl_im1k_lin834.yaml \
  --output-dir <PATH/TO/OUTPUT/DIR> \
  train.dataset_path=NCells:root=/<PATH/TO/CSV.GZ>:split=TRAIN \
  finetune.path='' \
  triplet.enable=true:weight_scaling=global \
  checkpointing.checkpointing_goal_epoch=40
```

Leave `finetune.path=''` to train from scratch.

### Fine-tuning from a Checkpoint

To resume from or fine-tune an existing checkpoint — for example, to apply the HDBSCAN contrastive loss on top of the Double Teacher checkpoint — set `finetune.path` to the checkpoint path:

```shell
PYTHONPATH=${PWD} python -m dinov3.run.submit dinov3/train/train.py \
  --nodes 1 \
  --config-file dinov3/configs/train/vitl_im1k_lin834.yaml \
  --output-dir <PATH/TO/OUTPUT/DIR> \
  train.dataset_path=NCells:root=/<PATH/TO/CSV.GZ>:split=TRAIN \
  finetune.path='<PATH/TO/CHECKPOINT>' \
  triplet.enable=true:weight_scaling=global \
  checkpointing.checkpointing_goal_epoch=40
```

---

## License

This project is released under the DINOv3 License. See [LICENSE.md](LICENSE.md) for full terms.

## Contributing

We welcome contributions. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). By participating, you agree to uphold these standards.

---

## Citing This Work

If you find this repository useful, please consider giving a star :star: and citing our paper:

```bibtex
@inproceedings{,
  title     = {Hierarchy-Aware Self-Supervised Learning for Biological Cell Images},
  author    = {},
  booktitle = {},
  year      = {2026},
}
```
