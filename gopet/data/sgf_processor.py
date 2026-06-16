"""SGF replay processor producing NumPy arrays for PyTorch training."""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DLGO_ROOT = ROOT / "DLGO_code"
if not DLGO_ROOT.exists():
    # Backwards compatibility with older folder name.
    DLGO_ROOT = ROOT / "DLGO - code"
if DLGO_ROOT.exists() and str(DLGO_ROOT) not in sys.path:
    sys.path.insert(0, str(DLGO_ROOT))

from dlgo.gosgf import Sgf_game

from gopet.board import FastBoard
from gopet.encoding import encode_planes_numpy, move_to_action
from gopet.game_state import GameState
from gopet.types import Color, Move


class GoDatasetBuilder:
    """Replay SGF games into feature/label arrays."""

    def __init__(
        self,
        board_size: int = 19,
        num_planes: int = 5,
        data_directory: str = "data",
    ) -> None:
        self.board_size = board_size
        self.num_planes = num_planes
        self.data_dir = Path(data_directory)

    @staticmethod
    def get_handicap(sgf: Sgf_game) -> Tuple[GameState, bool]:
        board = FastBoard(19, 19)
        first_move_done = False
        game_state = GameState.new_game(19)

        handicap = sgf.get_handicap()
        if handicap is not None and handicap != 0:
            for setup in sgf.get_root().get_setup_stones():
                for move in setup:
                    row, col = move
                    board.place_stone(Color.black, row, col)
            first_move_done = True
            game_state = GameState(board, Color.white, None, None)
        return game_state, first_move_done

    def replay_game(
        self,
        sgf_content: bytes,
    ) -> Tuple[np.ndarray, np.ndarray]:
        sgf = Sgf_game.from_string(sgf_content)
        game_state, first_move_done = self.get_handicap(sgf)

        features: List[np.ndarray] = []
        labels: List[int] = []

        for item in sgf.main_sequence_iter():
            color, move_tuple = item.get_move()
            if color is None:
                continue

            if move_tuple is not None:
                row, col = move_tuple
                move = Move.play(row, col)
            else:
                move = Move.pass_turn()

            if first_move_done and move.is_play:
                features.append(
                    encode_planes_numpy(game_state, num_planes=self.num_planes)
                )
                labels.append(move_to_action(game_state, move))

            game_state = game_state.apply_move(move)
            first_move_done = True

        if not features:
            return (
                np.zeros((0, self.num_planes, self.board_size, self.board_size), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
            )
        return np.stack(features).astype(np.float32), np.asarray(labels, dtype=np.int64)

    def process_sgf_files(
        self,
        sgf_paths: Sequence[str | Path],
        *,
        skip_errors: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        all_features: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []
        skipped = 0
        for path in sgf_paths:
            content = Path(path).read_bytes()
            try:
                features, labels = self.replay_game(content)
            except Exception:
                if not skip_errors:
                    raise
                skipped += 1
                continue
            if len(features):
                all_features.append(features)
                all_labels.append(labels)
        if skip_errors and skipped:
            print(f"Skipped {skipped} SGF files due to replay errors.", file=sys.stderr)
        if not all_features:
            empty_f = np.zeros(
                (0, self.num_planes, self.board_size, self.board_size),
                dtype=np.float32,
            )
            return empty_f, np.zeros((0,), dtype=np.int64)
        return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)

    def save(self, features: np.ndarray, labels: np.ndarray, prefix: str) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.data_dir / f"{prefix}_features.npy", features)
        np.save(self.data_dir / f"{prefix}_labels.npy", labels)


class SGFReplayDataset:
    """Minimal PyTorch-compatible dataset wrapper."""

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        return self.features[index], int(self.labels[index])

    @classmethod
    def from_npy(cls, data_dir: str, prefix: str) -> SGFReplayDataset:
        base = Path(data_dir)
        features = np.load(base / f"{prefix}_features.npy")
        labels = np.load(base / f"{prefix}_labels.npy")
        return cls(features, labels)

    def to_torch_dataset(self):
        import torch
        from torch.utils.data import TensorDataset

        x = torch.from_numpy(self.features)
        y = torch.from_numpy(self.labels).long()
        return TensorDataset(x, y)


def iter_sgf_files(directory: str | Path) -> Iterator[Path]:
    directory = Path(directory)
    yield from directory.rglob("*.sgf")


def build_from_directory(
    sgf_directory: str | Path,
    output_prefix: str = "train",
    data_directory: str = "data",
    num_planes: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    builder = GoDatasetBuilder(data_directory=data_directory, num_planes=num_planes)
    paths = list(iter_sgf_files(sgf_directory))
    features, labels = builder.process_sgf_files(paths)
    builder.save(features, labels, output_prefix)
    return features, labels
