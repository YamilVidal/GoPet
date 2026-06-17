"""Dataset building helpers for basic_cnn training."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from gopet.data.sgf_processor import GoDatasetBuilder, iter_sgf_files
from gopet.data.training_cache import SplitRecord, cache_is_valid, record_split


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
    dataset_id: Optional[str] = None,
    sgf_directory: Optional[str | Path] = None,
    max_games: Optional[int] = None,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    builder = GoDatasetBuilder(
        board_size=board_size,
        num_planes=num_planes,
        data_directory=str(data_directory),
    )
    features, labels = builder.process_sgf_files(
        sgf_paths,
        skip_errors=skip_errors,
        phase=f"preprocess {output_prefix}",
    )
    if save and len(features):
        print(f"Saving {output_prefix} cache to {data_directory} ...")
        builder.save(features, labels, output_prefix)
        if dataset_id is not None and sgf_directory is not None:
            record_split(
                data_directory,
                dataset_id=dataset_id,
                board_size=board_size,
                num_planes=num_planes,
                split=SplitRecord(
                    prefix=output_prefix,
                    sgf_directory=str(Path(sgf_directory).resolve()),
                    max_games=max_games,
                    seed=seed,
                    position_count=int(features.shape[0]),
                    sgf_file_count=len(sgf_paths),
                    features_file=f"{output_prefix}_features.npy",
                    labels_file=f"{output_prefix}_labels.npy",
                    board_size=board_size,
                    num_planes=num_planes,
                ),
            )
    return features, labels


def load_or_build_dataset(
    sgf_directory: str | Path,
    *,
    data_directory: str | Path,
    prefix: str,
    max_games: Optional[int] = None,
    seed: int = 0,
    board_size: int = 19,
    num_planes: int = 5,
    skip_errors: bool = True,
    force_rebuild: bool = False,
    dataset_id: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    data_dir = Path(data_directory)
    features_path = data_dir / f"{prefix}_features.npy"
    labels_path = data_dir / f"{prefix}_labels.npy"

    if not force_rebuild and cache_is_valid(
        data_dir,
        prefix=prefix,
        sgf_directory=sgf_directory,
        max_games=max_games,
        seed=seed,
        board_size=board_size,
        num_planes=num_planes,
    ):
        features = np.load(features_path)
        labels = np.load(labels_path)
        if features.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Cache shape mismatch for '{prefix}': "
                f"{features.shape[0]} features vs {labels.shape[0]} labels"
            )
        return features, labels

    paths = sample_sgf_paths(sgf_directory, max_games=max_games, seed=seed)
    if not paths:
        raise FileNotFoundError(f"No SGF files found under {sgf_directory}")

    print(f"Building cached dataset split '{prefix}' ({len(paths):,} SGF files)")

    try:
        features, labels = build_training_arrays(
            paths,
            data_directory=data_directory,
            output_prefix=prefix,
            num_planes=num_planes,
            board_size=board_size,
            skip_errors=skip_errors,
            save=True,
            dataset_id=dataset_id,
            sgf_directory=sgf_directory,
            max_games=max_games,
            seed=seed,
        )
    except Exception:
        for path in (features_path, labels_path):
            if path.exists():
                path.unlink()
        raise

    if len(features) == 0:
        for path in (features_path, labels_path):
            if path.exists():
                path.unlink()
        raise ValueError(f"No training positions produced for split '{prefix}'")

    return features, labels
