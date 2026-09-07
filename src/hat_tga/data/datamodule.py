"""PyTorch Lightning DataModule for wind super-resolution."""

import logging
from typing import Optional

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from hat_tga.data import DatasetFactory

logger = logging.getLogger(__name__)


class WindSRDataModule(pl.LightningDataModule):
    """DataModule wrapping WindSRDataset for all splits.

    Args:
        cfg: Full Hydra config (cfg.dataset, cfg.trainer used).
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._datasets: dict[str, object] = {}

    def setup(self, stage: Optional[str] = None) -> None:
        DatasetCls = DatasetFactory(self.cfg.dataset.name)

        splits = []
        if stage in ("fit", None):
            splits += ["train", "val"]
        if stage in ("validate", None):
            splits += ["val"]
        if stage in ("test", None):
            splits += ["test"]
        seen: set = set()
        splits = [s for s in splits if not (s in seen or seen.add(s))]

        for split in splits:
            if split not in self._datasets:
                ds_cfg = OmegaConf.to_container(self.cfg.dataset, resolve=True)
                ds_cfg["split"] = split
                self._datasets[split] = DatasetCls(OmegaConf.create(ds_cfg))

    def _make_loader(self, split: str, shuffle: bool) -> DataLoader:
        t = self.cfg.trainer
        return DataLoader(
            self._datasets[split],
            batch_size=t.batch_size,
            shuffle=shuffle,
            num_workers=t.get("num_workers", 4),
            pin_memory=t.get("pin_memory", True),
            drop_last=(split == "train"),
            persistent_workers=t.get("num_workers", 4) > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader("train", shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader("val", shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader("test", shuffle=False)
