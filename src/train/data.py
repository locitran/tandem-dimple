"""PyTorch dataloaders for R20000 feature tables."""

import os
import sys
from typing import Dict, Sequence

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

    def __init__(self, x, y, savs):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.savs = np.asarray(savs)

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
            "sav": str(self.savs[index])
        }
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
) -> Dict[str, object]:

    assert len(feat_names) <= input_dim, f"input_dim={input_dim} is smaller than the number of selected features ({len(feat_names)})."

    data = np.load(fold_path, allow_pickle=True)[f"fold{fold}"].item()
    df_feat = pd.read_csv(feat_path)
    mean, std = get_r20000_normalizer(feat_path=feat_path,feat_names=feat_names)

    df_by_sav = df_feat.set_index("SAV_coords", drop=False)
    split_data = {}
    for split in ["train", "val"]:
        savs = np.asarray(data[split])
        
        split_df = df_by_sav.loc[savs]
        split_data[split] = {
            "savs": savs,
            "x": split_df[feat_names].to_numpy(dtype=np.float32),
            "y": split_df["labels"].to_numpy(dtype=np.int64),
        }

    for split in ["train", "val"]:
        split_data[split]["x"] = _normalize_and_pad(split_data[split]["x"],mean,std,input_dim,)

    train_ds = R20000Dataset(split_data["train"]["x"], split_data["train"]["y"], split_data["train"]["savs"])
    val_ds = R20000Dataset(split_data["val"]["x"], split_data["val"]["y"], split_data["val"]["savs"])

    generator = torch.Generator()
    generator.manual_seed(seed)
    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=generator,),
        "val": DataLoader(val_ds,batch_size=batch_size,shuffle=False,num_workers=num_workers,),
    }


def _normalize_and_pad(x, mean, std, input_dim):
    x = x.copy()
    nan_rows, nan_cols = np.where(np.isnan(x))
    x[nan_rows, nan_cols] = mean[nan_cols]
    x = (x - mean) / std

    padded_x = np.zeros((x.shape[0], input_dim), dtype=np.float32)
    padded_x[:, : x.shape[1]] = x
    return padded_x


def get_r20000_normalizer(
    feat_path: str = TANDEM_R20000,
    feat_names: Sequence[str] = TANDEM_FEATS["v1.1"],
):
    """Return mean/std for the selected features using the full R20000 table."""
    df_feat = pd.read_csv(feat_path)
    x = df_feat[list(feat_names)].to_numpy(dtype=np.float32)

    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    std[std == 0] = 1.0
    return mean, std


def prepare_r20000_test_arrays(
    fold_path: str = R20000_CV_SAVS,
    feat_path: str = TANDEM_R20000,
    feat_names: Sequence[str] = TANDEM_FEATS["v1.1"],
    input_dim: int = 50,
    fold: int = 0,
):
    """Prepare the held-out R20000 test split as normalized arrays."""
    assert len(feat_names) <= input_dim, f"input_dim={input_dim} is smaller than the number of selected features ({len(feat_names)})."

    data = np.load(fold_path, allow_pickle=True)[f"fold{fold}"].item()
    df_feat = pd.read_csv(feat_path)
    df_by_sav = df_feat.set_index("SAV_coords", drop=False)

    savs = np.asarray(data["test"]).astype(str)
    test_df = df_by_sav.loc[savs]
    mean, std = get_r20000_normalizer(feat_path=feat_path, feat_names=feat_names)

    x = test_df[list(feat_names)].to_numpy(dtype=np.float32)
    x = _normalize_and_pad(x, mean, std, input_dim)
    y = test_df["labels"].to_numpy(dtype=np.int64)

    output = {
        "x": x,
        "y": y,
        "savs": savs,
        "dataframe": test_df.reset_index(drop=True),
    }
    if "test_SASA" in data:
        output["SASA"] = np.asarray(data["test_SASA"], dtype=np.float32)
    if "test_exposure_group" in data:
        output["exposure_group"] = np.asarray(data["test_exposure_group"]).astype(str)
    return output


def prepare_external_test_arrays(
    test_feat_path: str,
    r20000_feat_path: str = TANDEM_R20000,
    feat_names: Sequence[str] = TANDEM_FEATS["v1.1"],
    input_dim: int = 50,
    label_col: str = "labels",
):
    """Prepare an external test set, such as GJB2, for model prediction.

    The external feature table must contain ``SAV_coords`` and the selected
    ``feat_names``.

    Normalization always uses the full R20000 feature table. External/gene-
    specific test data should not fit its own mean and standard deviation.
    """
    assert len(feat_names) <= input_dim, f"input_dim={input_dim} is smaller than the number of selected features ({len(feat_names)})."

    df_test = pd.read_csv(test_feat_path)
    savs = df_test["SAV_coords"].to_numpy().astype(str)

    mean, std = get_r20000_normalizer(feat_path=r20000_feat_path,feat_names=feat_names,)
    x = df_test[list(feat_names)].to_numpy(dtype=np.float32)
    x = _normalize_and_pad(x, mean, std, input_dim)
    y = df_test[label_col].to_numpy(dtype=np.int64) if label_col in df_test.columns else None

    return {
        "x": x,
        "y": y,
        "savs": savs,
        "dataframe": df_test.reset_index(drop=True),
    }
