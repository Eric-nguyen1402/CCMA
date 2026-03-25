"""Wrapper functions to standardize vision datasets."""


import os
from typing import Optional

# from bcv.datasets.biology.icpr2020_pollen import ICPR2020Pollen
# from bcv.datasets.cell_biology.bbbc.bbbc048_cell_cycle import BBBC048CellCycleDataset
# from bcv.datasets.cell_biology.iicbu2008_hela import IICBU2008HeLa
# from bcv.datasets.cytology.blood_smear.acevedo_et_al_2020 import BloodSmearDataSet
# from bcv.datasets.cytology.blood_smear.malaria import MalariaDataset
# from bcv.datasets.cytology.pap_smear.hussain_et_al_2019 import Hussain2019Dataset
# from bcv.datasets.cytology.pap_smear.plissiti_et_al_2018 import Plissiti2018Dataset
# from bcv.datasets.dermoscopy.ham10000 import HAM10000Dataset
# from bcv.datasets.fundoscopy.diabetic_retinopathy import DiabeticRetinopathyDataset
# from bcv.datasets.pathology.amyloid_beta.tang_et_al_2019 import AmyloidBeta2019Dataset
# from bcv.datasets.pathology.idr0042_upenn_heart import UPennHeart2018Dataset
# from bcv.datasets.pathology.iicbu2008_lymphoma import IICBU2008Lymphoma
# from bcv.datasets.pathology.kather_et_al_2016 import ColorectalHistologyDataset
# from bcv.datasets.pathology.mhist import MHist
# from bcv.datasets.pathology.patch_camelyon import PatchCamelyonDataSet
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import DTD
from torchvision.datasets import SVHN
from torchvision.datasets import FGVCAircraft
from torchvision.datasets import Flowers102
from torchvision.datasets import Food101
from torchvision.datasets import ImageFolder
from torchvision.datasets import OxfordIIITPet
from torchvision.datasets import Places365
from torchvision.datasets import StanfordCars
from torchvision.datasets import VisionDataset
from torchvision.datasets import Caltech101
from torchvision.datasets import Caltech256
from torchvision.datasets import STL10
from torchvision.datasets import SUN397

from ALFM.src.datasets.utils import CustomImageFolder

import json
from torchvision import transforms


class Food101Wrapper:
    @staticmethod  # don't even ask
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: bool = False,
    ) -> VisionDataset:
        split = "train" if train else "test"
        return Food101(root, split, transform, download=download)


class StanfordCarsWrapper:
    @staticmethod
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: bool = False,
    ) -> VisionDataset:
        split = "train" if train else "test"
        return StanfordCars(root, split, transform, download=download)


class FGVCAircraftWrapper:
    @staticmethod
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: bool = False,
    ) -> VisionDataset:
        split = "trainval" if train else "test"
        return FGVCAircraft(root, split, transform=transform, download=download)


class DTDWrapper:
    @staticmethod
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: bool = False,
    ) -> VisionDataset:
        split = "train" if train else "test"
        return DTD(root, split, partition=1, transform=transform, download=False)

class STL10Wrapper:
    @staticmethod
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: bool = False,
    ) -> VisionDataset:
        split = "train" if train else "test"
        return STL10(root, split=split, transform=transform, download=download)


class Caltech101Wrapper:
    @staticmethod
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: bool = False,
    ) -> VisionDataset:
        """
        NOTE: Caltech101 does not have a standard train/test split.
        This wrapper returns the entire dataset regardless of the `train` flag.
        You must manually split it into train and validation sets.
        """
        if transform is None:
            transform = transforms.Compose([
                transforms.Lambda(lambda img: img.convert("RGB")),  # <-- Fix grayscale
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
            ])
        else:
            # Safely inject convert("RGB") before ToTensor if user provides a transform
            transform = transforms.Compose([
                transforms.Lambda(lambda img: img.convert("RGB"))
            ] + list(transform.transforms))
            
        return Caltech101(root, transform=transform, download=False)


class Caltech256Wrapper:
    @staticmethod
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: bool = False,
    ) -> VisionDataset:
        """
        NOTE: Caltech256 does not have a standard train/test split.
        This wrapper returns the entire dataset regardless of the `train` flag.
        You must manually split it into train and validation sets.
        """
        if transform is None:
            transform = transforms.Compose([
                transforms.Lambda(lambda img: img.convert("RGB")),  # <-- Fix grayscale
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
            ])
        else:
            # Safely inject convert("RGB") before ToTensor if user provides a transform
            transform = transforms.Compose([
                transforms.Lambda(lambda img: img.convert("RGB"))
            ] + list(transform.transforms))

        return Caltech256(root, transform=transform, download=False)


class SUN397Wrapper:
    @staticmethod
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: bool = False,
    ) -> VisionDataset:
        """
        NOTE: The torchvision SUN397 dataset does not expose a train/test split directly.
        This wrapper returns the full dataset for a given partition.
        You must manually split it using the partition files.
        """
        # The original paper defines 10 partitions for cross-validation.
        # Following DTD's example, we default to the first partition.
        return SUN397(root, transform=transform, download=False)

class OxfordIIITPetWrapper:
    @staticmethod
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: bool = False,
    ) -> VisionDataset:
        split = "trainval" if train else "test"
        return OxfordIIITPet(
            root, split, target_types="category", transform=transform, download=download
        )


class Flowers102Wrapper:
    @staticmethod
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: bool = False,
    ) -> VisionDataset:
        split = "train" if train else "test"
        return Flowers102(root, split, transform, download=download)


class SVHNWrapper:
    @staticmethod  # don't even ask
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: bool = False,
    ) -> VisionDataset:
        split = "train" if train else "test"
        return SVHN(root, split, transform, download=download)


class DomainNetRealWrapper:
    @staticmethod  # don't even ask
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: bool = False,
    ) -> VisionDataset:
        root = os.path.join(root, "domainnet_real")
        file = "real_train.txt" if train else "real_test.txt"
        file = os.path.join(root, file)
        return CustomImageFolder(root, file, transform=transform)


# class ImageNet100Wrapper:
#     @staticmethod  # don't even ask
#     def __call__(
#         root: str,
#         train: bool,
#         transform: Optional[transforms.Compose] = None,
#         download: bool = False,
#     ) -> VisionDataset:
#         split = "train" if train else "val"
#         root = os.path.join(root, "imagenet100", split)
#         return ImageFolder(root, transform=transform)

class ImageNet100Wrapper:
    @staticmethod
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: bool = False,
    ) -> VisionDataset:
        # Set split directory (train or val)
        split = "train" if train else "val"
        split_dir = os.path.join(root, "imagenet100", split)

        # Load base dataset using ImageFolder
        dataset = ImageFolder(split_dir, transform=transform)

        # Load label mapping: WordNet ID → readable name
        label_path = os.path.join(root, "imagenet100", "Labels.json")
        with open(label_path, "r") as f:
            wnid_to_name = json.load(f)

        # Map dataset.classes (wnids) to readable names
        readable_classes = []
        for wnid in dataset.classes:
            readable_name = wnid_to_name.get(wnid, wnid)  # fallback to wnid if not found
            readable_classes.append(readable_name)

        # Patch dataset to use readable names
        dataset.readable_classes = readable_classes           # custom attribute
        dataset.wnid_to_name = wnid_to_name                   # store full mapping
        dataset.class_to_idx_readable = {
            readable_name: dataset.class_to_idx[wnid]
            for wnid, readable_name in zip(dataset.classes, readable_classes)
        }

        # Optional: Overwrite `classes` if your downstream code uses it for prompts
        dataset.classes = readable_classes
        dataset.class_to_idx = dataset.class_to_idx_readable

        return dataset


class Places365Wrapper:
    @staticmethod  # don't even ask
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: bool = False,
    ) -> VisionDataset:
        split = "train-standard" if train else "val"
        if download:  # Check if image archive already extracted
            try:
                Places365(
                    root, split, small=True, transform=transform, download=download
                )
            except RuntimeError:
                download = False
        return Places365(
            root, split, small=True, transform=transform, download=download
        )


class AmyloidBetaBalancedWrapper:
    @staticmethod  # don't even ask
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: Optional[bool] = False,
    ) -> Dataset:
        split = "train+val" if train else "test"
        return AmyloidBeta2019Dataset(
            root,
            split=split,
            transform=transform,
            download=download,
            balance_classes="rand_under",
        )


class AmyloidBetaUnbalancedWrapper:
    @staticmethod  # don't even ask
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: Optional[bool] = False,
    ) -> Dataset:
        split = "train+val" if train else "test"
        return AmyloidBeta2019Dataset(
            root,
            split=split,
            transform=transform,
            download=download,
            balance_classes=None,
        )


class BloodSmearWrapper:
    @staticmethod  # don't even ask
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: Optional[bool] = False,
    ) -> Dataset:
        split = "train+val" if train else "test"
        return BloodSmearDataSet(
            root, split=split, transform=transform, download=download
        )


class CellCycleWrapper:
    @staticmethod  # don't even ask
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: Optional[bool] = False,
    ) -> Dataset:
        split = "train+val" if train else "test"
        return BBBC048CellCycleDataset(
            root, split=split, transform=transform, download=download
        )


class ColonPolypsWrapper:
    @staticmethod  # don't even ask
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: Optional[bool] = False,
    ) -> Dataset:
        split = "train+val" if train else "test"
        return MHist(root, split=split, transform=transform, download=download)


class ColorectalHistologyWrapper:
    @staticmethod  # don't even ask
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: Optional[bool] = False,
    ) -> Dataset:
        split = "train+val" if train else "test"
        return ColorectalHistologyDataset(
            root, split=split, transform=transform, download=download
        )


class DiabeticRetinopathyWrapper:
    @staticmethod  # don't even ask
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: Optional[bool] = False,
    ) -> Dataset:
        split = "train+val" if train else "test"
        return DiabeticRetinopathyDataset(
            root, split=split, transform=transform, download=download
        )


class HAM10000Wrapper:
    @staticmethod  # don't even ask
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: Optional[bool] = False,
    ) -> Dataset:
        split = "train+val" if train else "test"
        return HAM10000Dataset(
            root, split=split, transform=transform, download=download
        )


class HeartFailureWrapper:
    @staticmethod  # don't even ask
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: Optional[bool] = False,
    ) -> Dataset:
        split = "train+val" if train else "test"
        return UPennHeart2018Dataset(
            root, split=split, transform=transform, download=download
        )


class IICBU2008HeLaWrapper:
    @staticmethod  # don't even ask
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: Optional[bool] = False,
    ) -> Dataset:
        split = "train+val" if train else "test"
        return IICBU2008HeLa(root, split=split, transform=transform, download=download)


class IICBU2008LymphomaWrapper:
    @staticmethod  # don't even ask
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: Optional[bool] = False,
    ) -> Dataset:
        split = "train+val" if train else "test"
        return IICBU2008Lymphoma(
            root, split=split, transform=transform, download=download
        )


class MalariaDatasetWrapper:
    @staticmethod  # don't even ask
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: Optional[bool] = False,
    ) -> Dataset:
        split = "train+val" if train else "test"
        return MalariaDataset(root, split=split, transform=transform, download=download)


class PollenWrapper:
    @staticmethod  # don't even ask
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: Optional[bool] = False,
    ) -> Dataset:
        split = "train+val" if train else "test"
        return ICPR2020Pollen(root, split=split, transform=transform, download=download)


class PatchCamelyonWrapper:
    @staticmethod  # don't even ask
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: Optional[bool] = False,
    ) -> Dataset:
        split = "train" if train else "test"
        return PatchCamelyonDataSet(
            root, split=split, transform=transform, download=download
        )


class PapSmear2018Wrapper:
    @staticmethod  # don't even ask
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: Optional[bool] = False,
    ) -> Dataset:
        split = "train+val" if train else "test"
        return Plissiti2018Dataset(
            root, split=split, transform=transform, download=download
        )


class CUB200Wrapper:
    @staticmethod
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: bool = False,
    ) -> VisionDataset:
        # CUB-200-2011 dataset implementation
        # Since CUB-200 is not in torchvision, we'll use ImageFolder structure
        split = "train" if train else "test"
        dataset_path = os.path.join(root, "CUB_200_2011", split)
        if not os.path.exists(dataset_path):
            # Try alternative path structure
            dataset_path = os.path.join(root, f"CUB_200_2011_{split}")
            if not os.path.exists(dataset_path):
                # Use the root directory itself
                dataset_path = root
        return ImageFolder(dataset_path, transform=transform)


class INaturalistWrapper:
    @staticmethod
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: bool = False,
    ) -> VisionDataset:
        # iNaturalist dataset implementation using ImageFolder
        split = "train" if train else "val"
        dataset_path = os.path.join(root, split)
        if not os.path.exists(dataset_path):
            # Use the root directory itself
            dataset_path = root
        return ImageFolder(dataset_path, transform=transform)


class MalariaDatasetWrapper:
    @staticmethod
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: bool = False,
    ) -> VisionDataset:
        # Malaria dataset implementation using ImageFolder
        split = "train" if train else "test"
        dataset_path = os.path.join(root, split)
        if not os.path.exists(dataset_path):
            dataset_path = root
        return ImageFolder(dataset_path, transform=transform)


class PollenWrapper:
    @staticmethod
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: bool = False,
    ) -> VisionDataset:
        # Pollen dataset implementation using ImageFolder
        split = "train" if train else "test"
        dataset_path = os.path.join(root, split)
        if not os.path.exists(dataset_path):
            dataset_path = root
        return ImageFolder(dataset_path, transform=transform)


class PapSmear2019Wrapper:
    @staticmethod  # don't even ask
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: Optional[bool] = False,
    ) -> Dataset:
        split = "train+val" if train else "test"
        return Hussain2019Dataset(
            root, split=split, transform=transform, download=download
        )
