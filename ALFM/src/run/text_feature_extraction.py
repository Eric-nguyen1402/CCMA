"""Text feature extraction for class names using CLIP text encoder.

This mirrors the image feature extraction pipeline but encodes the dataset's
class names as text and stores the resulting embeddings in the same feature
cache file under the "text" split.
"""

import logging
import os
from typing import List, Sequence, Optional

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from omegaconf import DictConfig

from ALFM.src.datasets.factory import create_dataset
from ALFM.src.datasets.registry import DatasetType
from ALFM.src.models.registry import ModelType
from ALFM.src.run.feature_extraction import (
    check_existing_features,
    save_vectors,
)

import hashlib, json
def _sha1_list(xs: List[str]) -> str:
    h = hashlib.sha1()
    for s in xs:
        h.update(s.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()

def _clean_name(s: str) -> str:
    # minimal normalization for nicer prompts
    s = s.strip()
    s = s.replace("_", " ")
    return " ".join(s.split())  # collapse whitespace


def _load_mapping_txt(path: str) -> dict:
    """
    Supports simple 'key<TAB>value' per line, or 'key value...' (first token = key).
    Ignores empty/comment lines. Returns {key: value}.
    """
    mp = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            ln = line.strip()
            if not ln or ln.startswith("#"):
                continue
            # split on tab if present, else on whitespace
            if "\t" in ln:
                k, v = ln.split("\t", 1)
            else:
                toks = ln.split()
                k, v = toks[0], " ".join(toks[1:]) if len(toks) > 1 else toks[0]
            mp[k] = v
    return mp


def _maybe_map_names(names: List[str],
                     dataset_cfg: DictConfig) -> List[str]:
    """
    If the dataset exposes WNIDs or other codes, map them to human-friendly names
    using optional config:
      - dataset_cfg.wnid_to_name_json : path to a JSON {wnid: "human name"}
      - dataset_cfg.wnid_to_name_txt  : path to a TXT with 'wnid<TAB>name'
    """
    mapping = {}
    path_json = getattr(dataset_cfg, "wnid_to_name_json", None)
    path_txt  = getattr(dataset_cfg, "wnid_to_name_txt", None)
    try:
        if path_json and os.path.isfile(path_json):
            with open(path_json, "r", encoding="utf-8") as fh:
                mapping.update(json.load(fh))
        if path_txt and os.path.isfile(path_txt):
            mapping.update(_load_mapping_txt(path_txt))
    except Exception as e:
        logging.warning(f"[classnames] failed loading mapping file: {e}")

    if mapping:
        out = [_clean_name(mapping.get(n, n)) for n in names]
    else:
        out = [_clean_name(n) for n in names]
    return out


def _from_indexed_list(names_like: Sequence[str]) -> List[str]:
    # If already a list aligned to indices 0..C-1, just clean it
    return [_clean_name(str(x)) for x in list(names_like)]


def _from_class_to_idx(class_to_idx: dict) -> List[str]:
    # Sort by the integer index to preserve class id order
    pairs = sorted(class_to_idx.items(), key=lambda x: x[1])
    return [_clean_name(str(k)) for (k, _) in pairs]


def _get_class_names(dataset: object, dataset_cfg: Optional[DictConfig] = None) -> List[str]:
    """
    Robust extraction of class names for *custom datasets*.

    Priority:
      0) Explicit class list file in config (dataset_cfg.class_names_path)
      1) dataset.classes (list-like, aligned to indices)
      2) dataset.class_to_idx (dict: name -> idx)
      3) dataset.idx_to_class or dataset.idx_to_label (list-like or dict idx->name)
      4) dataset.categories / dataset.labels / dataset.label_names / dataset.classnames
      5) dataset.synsets (list of objects with .name or (.wnid, .name))
      6) dataset.wnids / dataset.wordnet_ids  (optionally map via config)
      7) dataset.samples tuple list [(path, class_idx), ...]  → infer by grouping
    """
    # 0) explicit class list file (one name per line)
    if dataset_cfg is not None:
        class_list_path = getattr(dataset_cfg, "class_names_path", None)
        if class_list_path and os.path.isfile(class_list_path):
            with open(class_list_path, "r", encoding="utf-8") as fh:
                names = [ _clean_name(ln.strip()) for ln in fh if ln.strip() ]
            if len(names) == len(set(names)) and len(names) > 0:
                logging.info(f"[classnames] loaded {len(names)} names from file: {class_list_path}")
                return _maybe_map_names(names, dataset_cfg)

    # 1) torchvision-style
    classes = getattr(dataset, "classes", None)
    if isinstance(classes, Sequence) and len(classes) > 0:
        return _maybe_map_names(_from_indexed_list(classes), dataset_cfg or DictConfig({}))

    # 2) class_to_idx dict
    class_to_idx = getattr(dataset, "class_to_idx", None)
    if isinstance(class_to_idx, dict) and len(class_to_idx) > 0:
        return _maybe_map_names(_from_class_to_idx(class_to_idx), dataset_cfg or DictConfig({}))

    # 3) idx_to_class / idx_to_label
    idx_to_class = getattr(dataset, "idx_to_class", None) or getattr(dataset, "idx_to_label", None)
    if isinstance(idx_to_class, dict) and len(idx_to_class) > 0:
        # Convert to dense list 0..C-1
        C = max(int(k) for k in idx_to_class.keys()) + 1
        names = [ _clean_name(str(idx_to_class[i])) for i in range(C) ]
        return _maybe_map_names(names, dataset_cfg or DictConfig({}))
    if isinstance(idx_to_class, Sequence) and len(idx_to_class) > 0:
        return _maybe_map_names(_from_indexed_list(idx_to_class), dataset_cfg or DictConfig({}))

    # 4) common custom attributes
    for attr in ("categories", "labels", "label_names", "classnames"):
        val = getattr(dataset, attr, None)
        if isinstance(val, Sequence) and len(val) > 0:
            return _maybe_map_names(_from_indexed_list(val), dataset_cfg or DictConfig({}))

    # 5) synsets
    synsets = getattr(dataset, "synsets", None)
    if isinstance(synsets, Sequence) and len(synsets) > 0:
        # try .name, else tuple (.wnid, .name)
        names = []
        for s in synsets:
            if hasattr(s, "name"):
                names.append(str(s.name))
            elif isinstance(s, (tuple, list)) and len(s) >= 2:
                names.append(str(s[1]))
            else:
                names.append(str(s))
        return _maybe_map_names(_from_indexed_list(names), dataset_cfg or DictConfig({}))

    # 6) wnids
    for attr in ("wnids", "wordnet_ids"):
        wnids = getattr(dataset, attr, None)
        if isinstance(wnids, Sequence) and len(wnids) > 0:
            return _maybe_map_names(_from_indexed_list(wnids), dataset_cfg or DictConfig({}))

    # 7) samples (like ImageFolder): infer names by inverting samples
    samples = getattr(dataset, "samples", None)
    if isinstance(samples, Sequence) and len(samples) > 0 and isinstance(samples[0], (tuple, list)) and len(samples[0]) >= 2:
        # Build idx -> name via root folder names if available
        # Try dataset.class_to_idx if present; else derive from paths.
        class_to_idx = getattr(dataset, "class_to_idx", None)
        if isinstance(class_to_idx, dict) and len(class_to_idx) > 0:
            names = _from_class_to_idx(class_to_idx)
            return _maybe_map_names(names, dataset_cfg or DictConfig({}))
        else:
            # Infer folder name for each class_idx from first occurrence
            first_by_idx = {}
            for path, idx in samples:
                if idx not in first_by_idx:
                    first_by_idx[idx] = os.path.basename(os.path.dirname(path))
            C = max(int(k) for k in first_by_idx.keys()) + 1
            names = [ _clean_name(first_by_idx[i]) for i in range(C) ]
            return _maybe_map_names(names, dataset_cfg or DictConfig({}))

    raise RuntimeError(
        "Could not infer class names from dataset. "
        "Provide `dataset_cfg.class_names_path` or implement one of: "
        "`classes`, `class_to_idx`, `idx_to_class`, `categories`, `label_names`, `synsets`, `wnids`."
    )


def _format_prompt(template: str, cls: str) -> str:
    """Format a single prompt template with the class name.

    Supports "{}" or "{label}" placeholders; falls back to appending the class.
    """
    if "{}" in template:
        return template.format(cls)
    if "{label}" in template:
        return template.replace("{label}", cls)
    # fallback: append class name
    return f"{template} {cls}".strip()


@torch.inference_mode()
def _encode_texts(model: open_clip.CLIP, texts: List[str], device: torch.device,
                  model_name: str) -> torch.Tensor:
    """
    Encode texts with the tokenizer that matches the checkpoint.
    """
    tokenizer = open_clip.get_tokenizer(model_name)
    tokenized = tokenizer(texts).to(device)
    model = model.to(device).eval()
    feats = model.encode_text(tokenized)
    return feats.float().detach()


def extract_text_features(
    dataset_cfg: DictConfig,
    model_cfg: DictConfig,
    dataset_dir: str,
    model_dir: str,
    feature_dir: str,
    templates: Optional[List[str]] = None,
) -> None:
    """Extract text features for dataset class names and save to HDF.

    Uses multiple natural-language templates per class and averages the text embeddings.
    Saves under group name "text" in the same feature file used for image features:
      <feature_dir>/<dataset>/<model>.hdf with datasets:
        - text/features: (C, D)
        - text/labels:   (C, 1) with labels 0..C-1
    """
    dataset_type = DatasetType[dataset_cfg.name]
    model_type = ModelType[model_cfg.name]

    clip_models = {
        ModelType.openai_vit_B16,
        ModelType.openclip_vit_B16,
        ModelType.openclip_vit_L14,
        ModelType.openclip_vit_H14,
        ModelType.openclip_vit_g14,
        ModelType.openclip_vit_G14,
    }
    if model_type not in clip_models:
        raise ValueError(
            f"Model {model_type.name} does not provide a text encoder. "
            "Please choose an OpenCLIP/OpenAI CLIP model."
        )

    # Default prompt templates if none provided
    if not templates or len(templates) == 0:
        templates = [
            "a photo of a {}",
            "a blurry photo of a {}",
            "a close-up photo of a {}",
            "a bright photo of a {}",
            "a cropped photo of a {}",
            "a clean photo of a {}",
            "a photo of the {}",
            "a photo of one {}",
            "a rendition of a {}",
            "a low resolution photo of a {}",
        ]

    dst_dir = os.path.join(feature_dir, f"{dataset_type.name}")
    vector_file = os.path.join(dst_dir, f"{model_type.name}.hdf")
    os.makedirs(dst_dir, exist_ok=True)

    split = "text"
    if check_existing_features(vector_file, split):
        logging.warn(
            f"{split} features have already been computed for the {dataset_type.name} "
            + f"dataset with the {model_type.name} model"
        )
        return

    # Create dataset (no transform needed) just to read class names
    dataset = create_dataset(dataset_type, root=dataset_dir, train=True, transform=None)
    print("\n==================== DEBUG DATASET ====================")
    print(f"[DEBUG] Total samples: {len(dataset)}")

    # If dataset has samples
    if hasattr(dataset, "samples"):
        print(f"[DEBUG] First 5 samples:")
        for s in dataset.samples[:5]:
            print("    ", s)

        labels = [label for _, label in dataset.samples]
        print(f"[DEBUG] Unique labels: {sorted(set(labels))[:20]}")
        print(f"[DEBUG] Num unique labels: {len(set(labels))}")

    print(f"[DEBUG] dataset.classes: {getattr(dataset, 'classes', 'MISSING')}")
    print(f"[DEBUG] dataset.class_to_idx: {getattr(dataset, 'class_to_idx', 'MISSING')}")
    print("=======================================================\n")

    try:
        class_names = _get_class_names(dataset, dataset_cfg)
        # Post-condition: sanity check class index alignment
        
        try:
            sample_indices = []

            if hasattr(dataset, "samples") and isinstance(dataset.samples, list):
                sample_indices = [label for _, label in dataset.samples]
            elif hasattr(dataset, "targets") and isinstance(dataset.targets, list):
                sample_indices = dataset.targets
            elif hasattr(dataset, "labels") and isinstance(dataset.labels, list):
                sample_indices = dataset.labels

            unique_labels = sorted(set(sample_indices))
            expected_labels = list(range(len(class_names)))

            if unique_labels == expected_labels:
                logging.info(f"[align-check] Class indices appear well-aligned: {unique_labels}")
            else:
                logging.warning(
                    f"[align-check] Class indices in dataset = {unique_labels}, expected {expected_labels}"
                )
                logging.warning("[align-check] Potential misalignment between dataset labels and class_names.")
        except Exception as e:
            logging.warning(f"[align-check] Skipped due to error: {e}")

    except RuntimeError:
        num_classes = int(getattr(dataset_cfg, "num_classes", 0))
        if num_classes <= 0:
            raise
        class_names = [f"class {i}" for i in range(num_classes)]

    # Build multi-template prompts per class
    texts: List[str] = []
    slices: List[slice] = []
    for name in class_names:
        start = len(texts)
        class_prompts = [_format_prompt(t, name) for t in templates]
        texts.extend(class_prompts)
        end = len(texts)
        slices.append(slice(start, end))  # range for this class

    # Build text encoder
    clip_model, _ = open_clip.create_model_from_pretrained(
        *model_type.value, cache_dir=model_dir
    )

    model_name, pretrained = model_type.value

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_feats = _encode_texts(clip_model, texts, device, model_name)  # (C*T, D) raw

    # Average per class then L2-normalize
    C = len(class_names)
    D = all_feats.shape[1]
    out = torch.zeros(C, D, dtype=torch.float32)
    for i, s in enumerate(slices):
        mean_feat = all_feats[s].mean(dim=0)
        out[i] = mean_feat
    out = F.normalize(out, dim=-1)

    if (
        (hasattr(dataset, "samples") and len(dataset.samples) > 10)
        or (hasattr(dataset, "targets") and isinstance(dataset.targets, (list, np.ndarray, torch.Tensor)) and len(dataset.targets) > 10)
        or (hasattr(dataset, "labels") and isinstance(dataset.labels, (list, np.ndarray, torch.Tensor)) and len(dataset.labels) > 10)
    ):
        try:
            from torchvision import transforms
            from PIL import Image

            dummy_tf = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                                    [0.26862954, 0.26130258, 0.27577711]),
            ])

            imgs, lbls = [], []
            if hasattr(dataset, "samples"):
                for path, label in dataset.samples[:100]:
                    img = Image.open(path).convert("RGB")
                    imgs.append(dummy_tf(img))
                    lbls.append(label)
            else:
                # fall back to raw data tensors (CIFAR/STL style)
                data, labels = dataset.data[:100], (dataset.targets if hasattr(dataset, "targets") else dataset.labels)[:100]
                from torchvision.transforms import ToPILImage
                to_pil = ToPILImage()
                for arr, label in zip(data, labels):
                    img = to_pil(torch.tensor(arr))
                    imgs.append(dummy_tf(img))
                    lbls.append(int(label))

            img_tensor = torch.stack(imgs).to(device)
            lbl_tensor = torch.tensor(lbls, device=device)

            with torch.no_grad():
                clip_model = clip_model.to(device).eval()
                image_feats = F.normalize(clip_model.encode_image(img_tensor), dim=1)
                text_feats = out.to(device)
                logits = image_feats @ text_feats.T
                acc = (logits.argmax(1) == lbl_tensor).float().mean().item()
                logging.info(f"[align-check] ZS acc on {len(lbls)} images: top1 = {acc:.3f}")
        except Exception as e:
            logging.warning(f"[align-check] ZS sanity skipped: {e}")

    features = out.cpu().numpy()
    labels = np.arange(len(class_names), dtype=np.int64).reshape(-1, 1)

    extra_meta = {
        "model_name": model_name,
        "pretrained": pretrained,
        "templates_sha1": _sha1_list(templates),
        "classnames_sha1": _sha1_list(class_names),
    }
    save_vectors(features, labels, vector_file, split, meta=extra_meta)
