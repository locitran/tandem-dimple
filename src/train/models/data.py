"""PyTorch dataloaders for TANDEM training datasets."""

import os
import sys
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from ..features import TANDEM_FEATS
    from ..utils.settings import CLUSTER, TANDEM_R20000
    from .process_data import getR20000
except ImportError:
    # Support running src/train/model.py directly as a script.
    package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if package_root not in sys.path:
        sys.path.insert(0, package_root)
    from src.features import TANDEM_FEATS
    from src.utils.settings import CLUSTER, TANDEM_R20000
    from src.train.process_data import getR20000


class R20000Dataset(Dataset):
    """Wrap R20000 numpy arrays as a PyTorch Dataset."""

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
    feat_path: str = TANDEM_R20000,
    clstr_path: str = CLUSTER,
    feat_names: Sequence[str] = TANDEM_FEATS["v1.1"],
    fold: int = 0,
    batch_size: int = 512,
    shuffle_train: bool = True,
    num_workers: int = 0,
    seed: int = 17,
) -> Dict[str, DataLoader]:
    """Load the first R20000 split by default and return PyTorch dataloaders.

    The split and preprocessing are delegated to ``process_data.getR20000`` so
    this dataloader stays consistent with the existing TANDEM training code.
    """
    folds, _, _, _ = getR20000(feat_path=feat_path,clstr_path=clstr_path,feat_names=feat_names,)
    data = folds[fold]

    # getR20000 returns one-hot labels, while PyTorch CrossEntropyLoss expects
    # class-index labels: [1, 0] -> 0 and [0, 1] -> 1.
    y_train = np.argmax(data["train"]["y"], axis=1)
    y_val = np.argmax(data["val"]["y"], axis=1)
    y_test = np.argmax(data["test"]["y"], axis=1)

    train_ds = R20000Dataset(data["train"]["x"], y_train, data["train"]["SAV_coords"],)
    val_ds = R20000Dataset(data["val"]["x"],y_val,data["val"]["SAV_coords"],)
    test_ds = R20000Dataset(data["test"]["x"],y_test,data["test"]["SAV_coords"],)

    generator = torch.Generator()
    generator.manual_seed(seed)
    loaders = {
        "train": DataLoader(train_ds,batch_size=batch_size,shuffle=shuffle_train,num_workers=num_workers,generator=generator,),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,),
    }

    return loaders
