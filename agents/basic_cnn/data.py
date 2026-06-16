"""Dataset building helpers for basic_cnn training."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from gopet.data.sgf_processor import GoDatasetBuilder, iter_sgf_files


def sample_sgf_paths(
    directory: str | Path,
    max_games: Optional[int] = None,
    seed: int = 0,
) -> list[Path]:
    paths = list(iter_sgf_files(directory))
    if not paths:
        return []
    if max_games is None or len(paths) <= max_games:
        return paths
    rng = random.Random(seed)
    rng.shuffle(paths)
    return paths[:max_games]


def build_training_arrays(
    sgf_paths: Sequence[str | Path],
    *,
    data_directory: str | Path,
    output_prefix: str,
    num_planes: int = 5,
    board_size: int = 19,
    skip_errors: bool = True,
    save: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    builder = GoDatasetBuilder(
        board_size=board_size,
        num_planes=num_planes,
        data_directory=str(data_directory),
    )
    features, labels = builder.process_sgf_files(sgf_paths, skip_errors=skip_errors)
    if save and len(features):
        builder.save(features, labels, output_prefix)
    return features, labels


def load_or_build_dataset(
    sgf_directory: str | Path,
    *,
    data_directory: str | Path,
    prefix: str,
    max_games: Optional[int] = None,
    seed: int = 0,
    skip_errors: bool = True,
    force_rebuild: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    data_dir = Path(data_directory)
    features_path = data_dir / f"{prefix}_features.npy"
    labels_path = data_dir / f"{prefix}_labels.npy"

    if not force_rebuild and features_path.exists() and labels_path.exists():
        return np.load(features_path), np.load(labels_path)

    paths = sample_sgf_paths(sgf_directory, max_games=max_games, seed=seed)
    if not paths:
        raise FileNotFoundError(f"No SGF files found under {sgf_directory}")

    return build_training_arrays(
        paths,
        data_directory=data_directory,
        output_prefix=prefix,
        skip_errors=skip_errors,
        save=True,
    )
