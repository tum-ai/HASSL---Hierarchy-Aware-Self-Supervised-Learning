import json
import os
import random
import torch
import numpy as np
import math
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union, Any, Tuple
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import platform
import gzip
import csv

from .decoders import NCellDecoder, TargetDecoder
from .extended import ExtendedVisionDataset

_ALL_DATASETS = [
    "N_BCCD",
    "N_CMP_15_17_and_TNBC","N_CoNIC","N_CryoNuSeg","N_DynamicNuclearNet",
    "N_IHC_TMA","N_MoNuSAC","N_MoNuSeg","N_Neurips","N_NuInsSeg","N_PanNuke",
    "N_Phenoplex","N_cyto2","N_databowl","N_iPSC_Morpologies","N_iPSC_QCData",
    "N_lynsec13","N_omnipose","N_tissuenet","N_yeaz","N_Helmholtz",
]

_PER_DATASET_LIMITS = {k: -1 for k in _ALL_DATASETS}

# ------------------------- low-level utils -------------------------

def _read_shape_area(npy_path: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Header-only read of .npy -> (H, W, H*W). Returns (None, None, None) on failure."""
    try:
        mm = np.load(npy_path, allow_pickle=False, mmap_mode="r")
        h, w = int(mm.shape[0]), int(mm.shape[1])
        return h, w, h * w
    except Exception:
        return None, None, None

# picklable worker
def _read_shape_area_job(img_path: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    return _read_shape_area(img_path)

def _mask_guess_paths(mask_dir: Path, stem: str) -> List[Path]:
    return [
        mask_dir / f"{stem}.npy",
        mask_dir / f"{stem}_mask.npy",
        mask_dir / f"{stem}_masks.npy",
        mask_dir / f"{stem}-mask.npy",
        mask_dir / f"{stem}-masks.npy",
    ]

def _to_tensor_img(arr: np.ndarray) -> torch.Tensor:
    if arr.ndim == 2:
        arr = arr[:, :, None]
    t = torch.from_numpy(arr)
    if t.dtype != torch.float32:
        t = t.to(torch.float32)
    if float(t.max()) > 1.0:
        t = t / 255.0
    return t.permute(2, 0, 1).contiguous()

def _to_tensor_mask(arr: Optional[np.ndarray]) -> Optional[torch.Tensor]:
    if arr is None:
        return None
    if arr.ndim > 2:
        arr = arr.squeeze()
    t = torch.from_numpy(arr)
    if t.dtype != torch.uint8:
        t = t.to(torch.uint8)
    return (t != 0).to(torch.uint8)

def _subset_size(raw, n: int) -> Optional[int]:
    """-1->n, 0<p<=1 -> ceil(p*n), int>=0 -> min(k,n), float>1 -> min(int(raw),n), None->None."""
    if raw is None:
        return None
    if isinstance(raw, (int, np.integer)):
        return n if raw < 0 else min(int(raw), n)
    if isinstance(raw, float):
        if raw < 0:
            return n
        if 0 < raw <= 1:
            return max(1, math.ceil(raw * n))
        return min(int(raw), n)
    return None


# ------------------------- robust path parsing -------------------------

def _iter_original_dirs(root: Path, split: str, allowed: Optional[set]) -> List[Tuple[Path, str]]:
    """
    Find every '.../<split>/**/original' under selected origins.
    Returns list: (original_dir_path, origin_name)
    """
    origins = [p for p in root.iterdir() if p.is_dir() and (allowed is None or p.name in allowed)]
    if allowed is not None:
        name2path = {p.name: p for p in origins}
        origins = [name2path[n] for n in allowed if n in name2path]

    out: List[Tuple[Path, str]] = []
    for origin in origins:
        for dirpath, _, _ in os.walk(origin):
            if os.path.basename(dirpath) != "original":
                continue
            parts = Path(dirpath).relative_to(root).parts
            if split in parts[:-1]:
                out.append((Path(dirpath), origin.name))
    return out

def _derive_label_from_path(img_path: str, root: Path, split: str, origin: str) -> str:
    """
    Parse labels from absolute img_path.
    Supports:
      origin/<labels...>/<split>/original/file.npy
      origin/<split>/<labels...>/original/file.npy
    Returns "" if no labels found.
    """
    parts = Path(img_path).resolve().relative_to(root).parts
    # locate origin
    try:
        i0 = parts.index(origin)
    except ValueError:
        return ""
    # locate split after origin
    j = None
    for k in range(i0 + 1, len(parts)):
        if parts[k] == str(split):
            j = k
            break
    if j is None:
        return ""
    # locate 'original' after split
    try:
        k_orig = parts.index("original", j + 1)
    except ValueError:
        return ""
    pre = list(parts[i0 + 1 : j])
    post = list(parts[j + 1 : k_orig])
    label_parts = pre + post
    return "__".join(label_parts) if label_parts else ""

def _load_mask_for_row(mask_dir: str, stem: str) -> Optional[np.ndarray]:
    """Try common filenames for a binary mask .npy; return HxW uint8 array or None."""
    for p in _mask_guess_paths(Path(mask_dir), stem):
        if p.exists():
            m = np.load(str(p), allow_pickle=False)
            if m.ndim > 2:
                m = np.squeeze(m)
            # ensure binary {0,1} uint8
            m = (m > 0).astype("uint8")
            return m
    return None
# ------------------------- 1) MANIFEST BUILDER (guaranteed labels) -------------------------

def build_manifest_csv(
    root: Union[str, Path],
    split: str,
    out_csv_gz: Union[str, Path],
    *,
    datasets: Optional[Union[str, List[str]]] = None,   # filter top-level origins (N_*)
    min_area_px: int = 0,                                # early filter by H*W
    workers: int = 16,                                   # parallel header reads
    force_threads: Optional[bool] = None,                # True -> ThreadPool; default True on macOS
    chunk_size: int = 100_000,
) -> None:
    """
    ONE-TIME scan -> compressed manifest CSV with correct labels for every file.
    Writes rows:
        img_path,origin,label,mask_dir,has_empty,stem,h,w,area
    - 'label' is derived per-file from path; "" only if there truly are no label folders.
    """
    root = Path(root).expanduser().resolve()
    allowed = None if datasets is None else ({datasets} if isinstance(datasets, str) else set(datasets))
    original_dirs = _iter_original_dirs(root, str(split), allowed)

    # Build candidate list
    candidates: List[tuple] = []
    for orig_dir, origin_name in original_dirs:
        with os.scandir(orig_dir) as it:
            for e in it:
                if not e.is_file() or not e.name.endswith(".npy"):
                    continue
                img_path = Path(e.path).resolve()
                stem = img_path.stem
                base_dir = img_path.parent.parent  # parent of 'original'
                mask_dir = base_dir / "mask"
                empty_mask_dir = base_dir / "empty_mask"
                has_empty = 1 if empty_mask_dir.exists() else 0
                label = _derive_label_from_path(str(img_path), root, str(split), origin_name)
                candidates.append((str(img_path), origin_name, label, str(mask_dir), has_empty, stem))

    # Choose executor type
    if force_threads is None:
        use_threads = (platform.system() == "Darwin")
    else:
        use_threads = bool(force_threads)
    Executor = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    map_kwargs = {} if use_threads else {"chunksize": 64}

    # Stream to gz CSV
    out_csv_gz = Path(out_csv_gz)
    out_csv_gz.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_csv_gz, "wt", newline="") as gz, Executor(max_workers=max(1, workers)) as ex:
        writer = csv.writer(gz)
        writer.writerow(["img_path", "origin", "label", "mask_dir", "has_empty", "stem", "h", "w", "area"])

        for start in range(0, len(candidates), chunk_size):
            batch = candidates[start:start + chunk_size]
            paths  = [t[0] for t in batch]
            metas  = [t[1:] for t in batch]  # (origin,label,mask_dir,has_empty,stem)

            for (h, w, area), (origin,label,mask_dir,has_empty,stem), path in zip(
                ex.map(_read_shape_area_job, paths, **map_kwargs), metas, paths
            ):
                if h is None or w is None or area is None:
                    continue
                if min_area_px > 0 and area < min_area_px:
                    continue
                # write exactly what we derived; empty string only when truly no label dirs
                writer.writerow([path, origin, label, mask_dir, has_empty, stem, h, w, area])

class _Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    def __str__(self) -> str:
        return self.value

class NCells(ExtendedVisionDataset):
    Split = Union[_Split]

    def __init__(
        self,
        *,
        split: "NCells.Split",
        root: Optional[str] = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            image_decoder=NCellDecoder,
            target_decoder=TargetDecoder,
        )
        self.split = split
        self.manifest_csv_gz = str(Path(root).expanduser())
        self.strict_match = True
        self.mmap_images = True
        self.return_paths = False
        self._rng = random.Random(42)

        if self.split == NCells.Split.TEST:
            flag = self.root.lower().__contains__("test")
            print("Setup test NCells dataset. Check if the test .csv.gz is provided as manifest file")
            if not flag:
                print(f"Warning: Manifest file path does not contain test flag")

        allowed = _ALL_DATASETS

        # Load manifest rows
        rows = []
        with gzip.open(self.manifest_csv_gz, "rt", newline="") as gz:
            reader = csv.DictReader(gz)
            for r in reader:
                origin = r["origin"]
                if allowed is not None and origin not in allowed:
                    continue
                area = int(r["area"]) if r["area"] else 0
                if area < 20*18:
                    continue
                rows.append((
                    r["img_path"],
                    origin,
                    r["label"],        # may be "" only when dataset actually has no label dirs
                    r["mask_dir"],
                    int(r["has_empty"]),
                    r["stem"],
                    int(r["h"]),
                    int(r["w"]),
                    area,
                ))

        if not rows:
            raise RuntimeError("Manifest filter produced 0 rows.")

        # Per-origin subsampling BEFORE storing
        by_origin: Dict[str, List[tuple]] = {}
        for t in rows:
            by_origin.setdefault(t[1], []).append(t)

        sel_rows: List[tuple] = []
        max_per_dataset = None
        for origin, items in by_origin.items():
            raw = _PER_DATASET_LIMITS.get(origin)
            if raw is None and max_per_dataset is not None:
                raw = max_per_dataset
            n_keep = _subset_size(raw, len(items))
            if n_keep is None or n_keep >= len(items):
                sel_rows.extend(items)
            elif n_keep > 0:
                items_copy = items[:]
                self._rng.shuffle(items_copy)
                sel_rows.extend(items_copy[:n_keep])

        if not sel_rows:
            raise RuntimeError("All rows filtered out by per_dataset_limits/max_per_dataset.")

        # Optional on-the-fly relabel for legacy CSVs
        root_for_relabel = None
        split_for_relabel = None
        if root_for_relabel is not None and split_for_relabel is not None:
            root_fix = Path(root_for_relabel).expanduser().resolve()
            fixed_rows = []
            for (img_path, origin, label, mask_dir, has_empty, stem, h, w, area) in sel_rows:
                if (label or "").strip() == "":
                    label = _derive_label_from_path(img_path, root_fix, split_for_relabel, origin)
                fixed_rows.append((img_path, origin, label, mask_dir, has_empty, stem, h, w, area))
            sel_rows = fixed_rows

        # store
        # idx: 0=img_path,1=origin,2=label,3=mask_dir,4=has_empty,5=stem,6=h,7=w,8=area
        self._rows = sel_rows

    def get_image_data(self, index: int) -> bytes:
        img_path = self._rows[index][0]
        mmap = "r" if self.mmap_images else None
        img = np.load(img_path, allow_pickle=False, mmap_mode=mmap)  # HxWx3 npy
        if img.ndim == 2:
            img = img[..., None]
        if img.shape[-1] == 1:
            img = np.repeat(img[..., :1], 3, axis=-1)
        # Convert floats -> uint8; clip safety
        if np.issubdtype(img.dtype, np.floating):
            arr = np.clip(img, 0.0, 1.0)
            arr = (arr * 255.0).round().astype("uint8")
        else:
            arr = img.astype("uint8", copy=False)
        # Ensure contiguous (PIL likes contiguous arrays)
        arr = np.ascontiguousarray(arr)
        return Image.fromarray(arr)  # mode inferred (RGB)

    def get_target(self, index: int) -> str:
        label = self._rows[index][2]
        if label == "":
            label = self._rows[index][1]
        return label

    def get_segmentation_mask_of_image(self, index: int) -> Image.Image:
        """
        Returns the mask as a PIL Image (3-channel, uint8, same shape as get_image_data).
        """
        img_path, origin, label, mask_dir, has_empty, stem, h, w, area = self._rows[index]
        m = _load_mask_for_row(mask_dir, stem)
        if m is None:
            m = np.zeros((h, w), dtype="uint8")
        if m.shape != (h, w):
            m = np.array(Image.fromarray(m, mode="L").resize((w, h), Image.NEAREST), dtype="uint8")
        arr = np.stack([m, m, m], axis=-1).astype("uint8")  # 3-channel
        arr *= 255
        return Image.fromarray(arr)

    def __getitem__(self, index: int):
        img = self.get_image_data(index)                   # PIL Image (RGB)
        seg = self.get_segmentation_mask_of_image(index)   # PIL Image (3-ch uint8 mask)
        target = self.get_target(index)

        if self.transform is not None:
            # IMPORTANT: pass (img, seg) together so augmentation can do paired crops
            img = self.transform((img, seg))
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
    
    def __len__(self) -> int:
        return len(self._rows)