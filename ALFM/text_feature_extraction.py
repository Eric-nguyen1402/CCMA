"""Script to extract and save text features (class embeddings) using CLIP text encoder."""

import os

import hydra
from dotenv import dotenv_values
from omegaconf import DictConfig

from ALFM.src.run.text_feature_extraction import extract_text_features


os.environ["SLURM_JOB_NAME"] = "interactive"


@hydra.main(
    config_path="conf",
    config_name="text_feature_extraction.yaml",
    version_base="1.1",
)
def main(cfg: DictConfig) -> None:
    """Extract and save text features for dataset class names.

    This mirrors the image feature extraction CLI but encodes the dataset's
    class names via the model's text encoder and stores them under the
    "text" group in the same feature cache file.
    """
    env = dotenv_values()
    dataset_dir = env.get("DATASET_DIR", None)
    model_dir = env.get("MODEL_CACHE_DIR", None)
    feature_dir = env.get("FEATURE_CACHE_DIR", None)

    assert (
        dataset_dir is not None
    ), "Please set the 'DATASET_DIR' variable in your .env file"

    assert (
        model_dir is not None
    ), "Please set the 'MODEL_CACHE_DIR' variable in your .env file"

    assert (
        feature_dir is not None
    ), "Please set the 'FEATURE_CACHE_DIR' variable in your .env file"

    extract_text_features(
        cfg.dataset,
        cfg.model,
        dataset_dir,
        model_dir,
        feature_dir,
    )


if __name__ == "__main__":
    main()  # type: ignore[misc]
