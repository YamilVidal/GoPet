"""Region-filtered SGF dataset building for cnn_triad."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DLGO_ROOT = ROOT / "DLGO_code"
if not DLGO_ROOT.exists():
    DLGO_ROOT = ROOT / "DLGO - code"
if DLGO_ROOT.exists() and str(DLGO_ROOT) not in sys.path:
    sys.path.insert(0, str(DLGO_ROOT))

from dlgo.gosgf import Sgf_game

from agents.cnn_triad.regions import (
    TRIAD_BOARD_SIZE,
    Region,
    extract_crop,
    find_crop_for_move,
    global_move_to_local_action,
    head_shapes,
    list_crops,
    pass_action_index,
    require_triad_board_size,
)
from gopet.data.sgf_processor import GoDatasetBuilder, iter_sgf_files
from gopet.data.training_cache import SplitRecord, cache_is_valid, record_split
from gopet.encoding import encode_planes_numpy
from gopet.game_state import GameState
from gopet.progress import finish_progress_line, format_duration_hms, report_item_progress
from gopet.types import Move

HEAD_PREFIX = {
    Region.CORNER: "corner",
    Region.SIDE: "side",
    Region.CENTER: "center",
}


def head_cache_prefix(head: Region, split: str) -> str:
    return f"{HEAD_PREFIX[head]}_{split}"


class TriadDatasetBuilder:
    """Replay SGF games into per-head regional feature/label arrays."""

    def __init__(
        self,
        board_size: int = TRIAD_BOARD_SIZE,
        num_planes: int = 5,
        data_directory: str | Path = "data",
    ) -> None:
        require_triad_board_size(board_size)
        self.board_size = board_size
        self.num_planes = num_planes
        self.data_dir = Path(data_directory)
        self._shapes = head_shapes()

    def replay_game(self, sgf_content: bytes) -> Dict[Region, Tuple[np.ndarray, np.ndarray]]:
        sgf = Sgf_game.from_string(sgf_content)
        game_state, first_move_done = GoDatasetBuilder.get_handicap(sgf, self.board_size)

        buckets: Dict[Region, Tuple[List[np.ndarray], List[int]]] = {
            Region.CORNER: ([], []),
            Region.SIDE: ([], []),
            Region.CENTER: ([], []),
        }

        for item in sgf.main_sequence_iter():
            color, move_tuple = item.get_move()
            if color is None:
                continue

            if move_tuple is not None:
                row, col = move_tuple
                move = Move.play(row, col)
            else:
                move = Move.pass_turn()

            if first_move_done:
                planes = encode_planes_numpy(game_state, num_planes=self.num_planes)
                if move.is_pass:
                    for region in (Region.CORNER, Region.SIDE, Region.CENTER):
                        spec = list_crops(region)[0]
                        crop = extract_crop(planes, spec)
                        label = pass_action_index(spec.out_height, spec.out_width)
                        buckets[region][0].append(crop)
                        buckets[region][1].append(label)
                elif move.is_play:
                    spec = find_crop_for_move(move, self.board_size)
                    if spec is not None:
                        crop = extract_crop(planes, spec)
                        label = global_move_to_local_action(move, spec)
                        buckets[spec.region][0].append(crop)
                        buckets[spec.region][1].append(label)

            game_state = game_state.apply_move(move)
            first_move_done = True

        out: Dict[Region, Tuple[np.ndarray, np.ndarray]] = {}
        for region, (height, width) in self._shapes.items():
            features_list, labels_list = buckets[region]
            if not features_list:
                out[region] = (
                    np.zeros((0, self.num_planes, height, width), dtype=np.float32),
                    np.zeros((0,), dtype=np.int64),
                )
            else:
                out[region] = (
                    np.stack(features_list).astype(np.float32),
                    np.asarray(labels_list, dtype=np.int64),
                )
        return out

    def process_sgf_files(
        self,
        sgf_paths: Sequence[str | Path],
        *,
        skip_errors: bool = False,
        show_progress: bool = True,
        phase: str = "preprocess",
    ) -> Dict[Region, Tuple[np.ndarray, np.ndarray]]:
        merged: Dict[Region, List[np.ndarray]] = {r: [] for r in HEAD_PREFIX}
        merged_labels: Dict[Region, List[np.ndarray]] = {r: [] for r in HEAD_PREFIX}
        skipped = 0
        num_files = len(sgf_paths)
        next_progress = 5
        phase_start = time.time()

        if show_progress and num_files > 0:
            print(f"{phase}: replaying {num_files:,} SGF files (triad regional split)")

        for file_index, path in enumerate(sgf_paths, start=1):
            content = Path(path).read_bytes()
            try:
                per_game = self.replay_game(content)
            except Exception:
                if not skip_errors:
                    raise
                skipped += 1
                if show_progress:
                    next_progress = report_item_progress(
                        file_index,
                        num_files,
                        phase=phase,
                        next_progress=next_progress,
                        phase_start=phase_start,
                    )
                continue

            for region, (features, labels) in per_game.items():
                if len(features):
                    merged[region].append(features)
                    merged_labels[region].append(labels)

            if show_progress:
                next_progress = report_item_progress(
                    file_index,
                    num_files,
                    phase=phase,
                    next_progress=next_progress,
                    phase_start=phase_start,
                )

        finish_progress_line(num_files, show_progress=show_progress)

        if skip_errors and skipped:
            print(f"Skipped {skipped} SGF files due to replay errors.", file=sys.stderr)

        out: Dict[Region, Tuple[np.ndarray, np.ndarray]] = {}
        for region, (height, width) in self._shapes.items():
            if not merged[region]:
                out[region] = (
                    np.zeros((0, self.num_planes, height, width), dtype=np.float32),
                    np.zeros((0,), dtype=np.int64),
                )
            else:
                out[region] = (
                    np.concatenate(merged[region], axis=0),
                    np.concatenate(merged_labels[region], axis=0),
                )
        return out

    def save_head(
        self,
        region: Region,
        features: np.ndarray,
        labels: np.ndarray,
        prefix: str,
    ) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.data_dir / f"{prefix}_features.npy", features)
        np.save(self.data_dir / f"{prefix}_labels.npy", labels)


def sample_sgf_paths(
    directory: str | Path,
    max_games: Optional[int] = None,
    seed: int = 0,
) -> list[Path]:
    import random

    paths = list(iter_sgf_files(directory))
    if not paths:
        return []
    if max_games is None or len(paths) <= max_games:
        return paths
    rng = random.Random(seed)
    rng.shuffle(paths)
    return paths[:max_games]


def load_or_build_head_dataset(
    sgf_directory: str | Path,
    *,
    region: Region,
    data_directory: str | Path,
    split: str,
    max_games: Optional[int] = None,
    seed: int = 0,
    board_size: int = TRIAD_BOARD_SIZE,
    num_planes: int = 5,
    skip_errors: bool = True,
    force_rebuild: bool = False,
    dataset_id: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    require_triad_board_size(board_size)
    data_dir = Path(data_directory)
    prefix = head_cache_prefix(region, split)
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
        return np.load(features_path), np.load(labels_path)

    paths = sample_sgf_paths(sgf_directory, max_games=max_games, seed=seed)
    if not paths:
        raise FileNotFoundError(f"No SGF files found under {sgf_directory}")

    print(f"Building triad cache '{prefix}' ({len(paths):,} SGF files)")
    builder = TriadDatasetBuilder(
        board_size=board_size,
        num_planes=num_planes,
        data_directory=data_dir,
    )
    try:
        all_heads = builder.process_sgf_files(
            paths,
            skip_errors=skip_errors,
            phase=f"preprocess {prefix}",
        )
    except Exception:
        for path in (features_path, labels_path):
            if path.exists():
                path.unlink()
        raise

    features, labels = all_heads[region]
    if len(features) == 0:
        for path in (features_path, labels_path):
            if path.exists():
                path.unlink()
        raise ValueError(f"No training positions produced for head '{prefix}'")

    builder.save_head(region, features, labels, prefix)
    if dataset_id is not None:
        height, width = head_shapes()[region]
        record_split(
            data_dir,
            dataset_id=dataset_id,
            board_size=board_size,
            num_planes=num_planes,
            split=SplitRecord(
                prefix=prefix,
                sgf_directory=str(Path(sgf_directory).resolve()),
                max_games=max_games,
                seed=seed,
                position_count=int(features.shape[0]),
                sgf_file_count=len(paths),
                features_file=f"{prefix}_features.npy",
                labels_file=f"{prefix}_labels.npy",
                board_size=board_size,
                num_planes=num_planes,
            ),
        )
    return features, labels


def build_all_heads(
    sgf_directory: str | Path,
    *,
    data_directory: str | Path,
    split: str,
    max_games: Optional[int] = None,
    seed: int = 0,
    board_size: int = TRIAD_BOARD_SIZE,
    num_planes: int = 5,
    skip_errors: bool = True,
    force_rebuild: bool = False,
    dataset_id: Optional[str] = None,
) -> Dict[Region, Tuple[np.ndarray, np.ndarray]]:
    """Build all three head caches from one SGF pass."""
    require_triad_board_size(board_size)
    data_dir = Path(data_directory)
    paths = sample_sgf_paths(sgf_directory, max_games=max_games, seed=seed)
    if not paths:
        raise FileNotFoundError(f"No SGF files found under {sgf_directory}")

    builder = TriadDatasetBuilder(
        board_size=board_size,
        num_planes=num_planes,
        data_directory=data_dir,
    )
    all_heads = builder.process_sgf_files(
        paths,
        skip_errors=skip_errors,
        phase=f"preprocess triad {split}",
    )

    out: Dict[Region, Tuple[np.ndarray, np.ndarray]] = {}
    for region in (Region.CORNER, Region.SIDE, Region.CENTER):
        prefix = head_cache_prefix(region, split)
        features, labels = all_heads[region]
        if len(features) == 0:
            raise ValueError(f"No training positions produced for head '{prefix}'")
        builder.save_head(region, features, labels, prefix)
        if dataset_id is not None:
            record_split(
                data_dir,
                dataset_id=dataset_id,
                board_size=board_size,
                num_planes=num_planes,
                split=SplitRecord(
                    prefix=prefix,
                    sgf_directory=str(Path(sgf_directory).resolve()),
                    max_games=max_games,
                    seed=seed,
                    position_count=int(features.shape[0]),
                    sgf_file_count=len(paths),
                    features_file=f"{prefix}_features.npy",
                    labels_file=f"{prefix}_labels.npy",
                    board_size=board_size,
                    num_planes=num_planes,
                ),
            )
        out[region] = (features, labels)
    return out
