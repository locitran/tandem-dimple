"""PyTorch dataloaders for R20000 feature tables."""

import os
import sys
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from ..features import TANDEM_FEATS
    from ..utils.settings import ROOT_DIR, TANDEM_R20000
except ImportError:
    # Support running src/train/model.py directly as a script.
    package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if package_root not in sys.path:
        sys.path.insert(0, package_root)
    from src.features import TANDEM_FEATS
    from src.utils.settings import ROOT_DIR, TANDEM_R20000


R20000_CV_SAVS = os.path.join(ROOT_DIR, "data", "R20000", "R20000_5fold_CV.npz")


class R20000Dataset(Dataset):
    """Wrap feature arrays, labels, and SAV names as a PyTorch Dataset."""

    def __init__(self, x, y, savs: Optional[Sequence[str]] = None):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.savs = None if savs is None else np.asarray(savs)

        if len(self.x) != len(self.y):
            raise ValueError("x and y must contain the same number of samples.")
        if self.savs is not None and len(self.savs) != len(self.x):
            raise ValueError("savs must contain the same number of samples as x.")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        item = {
            "x": self.x[index],
            "y": self.y[index],
        }
        if self.savs is not None:
            item["sav"] = str(self.savs[index])
        return item


def get_r20000_dataloaders(
    fold_path: str = R20000_CV_SAVS,
    feat_path: str = TANDEM_R20000,
    feat_names: Sequence[str] = TANDEM_FEATS["v1.1"],
    input_dim: int = 50,
    fold: int = 0,
    batch_size: int = 512,
    num_workers: int = 0,
    seed: int = 17,
) -> Dict[str, DataLoader]:
    """Create train/val/test dataloaders from saved R20000 fold SAVs.

    Normalization is fitted on the training split only. The same train mean and
    standard deviation are then applied to validation and test splits.

    If input_dim is given, selected features are placed in the first
    columns and the remaining columns are filled with zeros. This is useful for
    fair ablation experiments where every model should receive the same input
    dimension even when only a few features are active.
    """
    if len(feat_names) > input_dim:
        raise ValueError(
            f"input_dim={input_dim} is smaller than the number of "
            f"selected features ({len(feat_names)})."
        )

    data = np.load(fold_path, allow_pickle=True)[f"fold{fold}"].item()
    df_feat = pd.read_csv(feat_path)

    df_by_sav = df_feat.set_index("SAV_coords", drop=False)
    split_data = {}
    for split in ["train", "val", "test"]:
        savs = np.asarray(data[split])
        
        split_df = df_by_sav.loc[savs]
        split_data[split] = {
            "savs": savs,
            "x": split_df[feat_names].to_numpy(dtype=np.float32),
            "y": split_df["labels"].to_numpy(dtype=np.int64),
        }

    train_mean = np.nanmean(split_data["train"]["x"], axis=0)
    train_std = np.nanstd(split_data["train"]["x"], axis=0)
    train_std[train_std == 0] = 1.0

    for split in ["train", "val", "test"]:
        x = split_data[split]["x"].copy()
        nan_rows, nan_cols = np.where(np.isnan(x))
        x[nan_rows, nan_cols] = train_mean[nan_cols]
        x = (x - train_mean) / train_std

        padded_x = np.zeros((x.shape[0], input_dim), dtype=np.float32)
        padded_x[:, : x.shape[1]] = x
        x = padded_x

        split_data[split]["x"] = x

    train_ds = R20000Dataset(split_data["train"]["x"], split_data["train"]["y"], split_data["train"]["savs"])
    val_ds = R20000Dataset(split_data["val"]["x"], split_data["val"]["y"], split_data["val"]["savs"])
    test_ds = R20000Dataset(split_data["test"]["x"], split_data["test"]["y"], split_data["test"]["savs"])

    generator = torch.Generator()
    generator.manual_seed(seed)
    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=generator,),
        "val": DataLoader(val_ds,batch_size=batch_size,shuffle=False,num_workers=num_workers,),
        "test": DataLoader(test_ds,batch_size=batch_size,shuffle=False,num_workers=num_workers,),
    }
    return loaders
