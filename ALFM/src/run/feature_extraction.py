"""Feature extraction and caching using pretraing backbones."""

import logging
import os
import warnings
from typing import Any
from typing import Callable

import h5py
import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from numpy.typing import NDArray
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from ALFM.src.datasets.factory import create_dataset
from ALFM.src.datasets.registry import DatasetType
from ALFM.src.models.backbone_wrapper import BackboneWrapper
from ALFM.src.models.factory import create_model
from ALFM.src.models.registry import ModelType
from ALFM.src.run.utils import SharedMemoryWriter

warnings.simplefilter("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
torch.set_float32_matmul_precision("medium")  # type: ignore[no-untyped-call]

import json


def check_existing_features(vector_file: str, split: str) -> bool:
    """Check if the features for the specified split already exist.

    Args:
        vector_file (str): Path to the HDF file containing the features.
        split (str): Split name, either 'train' or 'test'.

    Raises:
        RuntimeError: If the features for the specified split already exist in the HDF file.
    """
    if os.path.exists(vector_file):
        with h5py.File(vector_file, "r") as fh:
            return split in fh.keys()

    return False

def save_vectors(
    features: NDArray[np.float32],
    labels: NDArray[np.int64],
    vector_file: str,
    split: str,
    meta: dict | None = None,   # <-- NEW
) -> None:
    """
    Save features and labels. Optionally attach 'meta/*' fields for provenance.
    Expected meta keys (strings): 'model_name', 'pretrained',
                                  'templates_sha1', 'classnames_sha1', ...
    """
    with h5py.File(vector_file, "a") as fh:
        grp = fh.require_group(split)
        # overwrite if exists to avoid stale caches
        for k in ("features", "labels"):
            if k in grp: del grp[k]
        grp.create_dataset("features", data=features)
        grp.create_dataset("labels", data=labels)

        if meta:
            mgrp = fh.require_group("meta")
            str_dt = h5py.string_dtype(encoding="utf-8")
            for k, v in meta.items():
                # store everything as UTF-8 strings for portability
                if isinstance(v, (dict, list)):
                    v = json.dumps(v, ensure_ascii=False)
                v = str(v)
                if k in mgrp:
                    del mgrp[k]
                mgrp.create_dataset(k, data=v, dtype=str_dt)


def extract_features(
    dataset_cfg: DictConfig,
    train: bool,
    model_cfg: DictConfig,
    dataset_dir: str,
    model_dir: str,
    feature_dir: str,
    dataloader: Callable[..., DataLoader[Any]],
    trainer_cfg: DictConfig,
) -> None:
    """Extract features from the dataset using the specified pretrained model and save them to disk.

    Args:
        dataset_cfg (DictConfig): Config representing the dataset to use.
        train (bool): True if extracting features for the training set, False for the test set.
        model_cfg (DictConfig): Config representing the pretrained model to use.
        dataset_dir (str): Path to the directory containing the dataset.
        model_dir (str): Path to the directory containing the model cache.
        feature_dir (str): Path to the directory where extracted features will be saved.
        dataloader (Callable[..., DataLoader[Any]]): Callable to create DataLoader for the dataset.
        trainer_cfg (pl.Trainer): PyTorch Lightning Trainer config for feature extraction.
    """
    dataset_type = DatasetType[dataset_cfg.name]
    model_type = ModelType[model_cfg.name]

    split = "train" if train else "test"
    dst_dir = os.path.join(feature_dir, f"{dataset_type.name}")
    vector_file = os.path.join(dst_dir, f"{model_type.name}.hdf")
    os.makedirs(dst_dir, exist_ok=True)

    if check_existing_features(vector_file, split):
        logging.warn(
            f"{split} features have already been computed for the {dataset_type.name}"
            + f" dataset with the {model_type.name} model"
        )
        return  # skip feature extraction

    model, transform = create_model(model_type, cache_dir=model_dir)
    dataset = create_dataset(
        dataset_type, root=dataset_dir, train=train, transform=transform
    )

    model = BackboneWrapper(model)
    shm_writer = SharedMemoryWriter(
        len(dataset), dataset_cfg.num_classes, model_cfg.num_features
    )

    model_name, pretrained = model_type.value

    prog_bar = pl.callbacks.RichProgressBar()
    trainer = hydra.utils.instantiate(trainer_cfg, callbacks=[shm_writer, prog_bar])
    trainer.predict(model, dataloader(dataset), return_predictions=False)
    trainer.strategy.barrier()

    if trainer.local_rank == 0:
        features, labels = shm_writer.get_predictions()
        save_vectors(features, labels, vector_file, split, meta={
            "model_name": model_name,
            "pretrained": pretrained
        })
        shm_writer.close()

    trainer.strategy.barrier()
