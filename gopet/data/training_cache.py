"""Shared training-data cache layout and agent usage tracking."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
TRAINING_DATA_ROOT = ROOT / "training_data"
DATASETS_ROOT = TRAINING_DATA_ROOT / "datasets"
USAGE_ROOT = TRAINING_DATA_ROOT / "usage"
MANIFEST_FILENAME = "manifest.json"


@dataclass
class SplitRecord:
    """Metadata for one train/val shard in a shared dataset."""

    prefix: str
    sgf_directory: str
    max_games: Optional[int]
    seed: int
    position_count: int
    sgf_file_count: int
    features_file: str
    labels_file: str
    board_size: int = 19
    num_planes: int = 5

    def matches_request(
        self,
        *,
        sgf_directory: str | Path,
        max_games: Optional[int],
        seed: int,
        board_size: int,
        num_planes: int,
    ) -> bool:
        return (
            self.sgf_directory == str(Path(sgf_directory).resolve())
            and self.max_games == max_games
            and self.seed == seed
            and self.board_size == board_size
            and self.num_planes == num_planes
        )


@dataclass
class DatasetManifest:
    dataset_id: str
    created_at: str
    board_size: int
    num_planes: int
    splits: Dict[str, SplitRecord] = field(default_factory=dict)
    used_by: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> DatasetManifest:
        raw = json.loads(path.read_text(encoding="utf-8"))
        splits = {
            name: SplitRecord(**split_data)
            for name, split_data in raw.get("splits", {}).items()
        }
        return cls(
            dataset_id=raw["dataset_id"],
            created_at=raw["created_at"],
            board_size=raw["board_size"],
            num_planes=raw["num_planes"],
            splits=splits,
            used_by=list(raw.get("used_by", [])),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset_id": self.dataset_id,
            "created_at": self.created_at,
            "board_size": self.board_size,
            "num_planes": self.num_planes,
            "splits": {name: asdict(split) for name, split in self.splits.items()},
            "used_by": self.used_by,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def split_record(self, prefix: str) -> Optional[SplitRecord]:
        return self.splits.get(prefix)


def slugify_sgf_directory(sgf_directory: str | Path) -> str:
    path = Path(sgf_directory)
    parts = [part for part in path.parts if part not in (".", "..")]
    if len(parts) >= 2:
        return f"{parts[-2]}_{parts[-1]}"
    if parts:
        return parts[-1]
    return "unknown"


def make_dataset_id(
    *,
    train_sgf_directory: str | Path,
    max_games: int,
    seed: int,
    board_size: int = 19,
    num_planes: int = 5,
    test_mode: bool = False,
) -> str:
    source = slugify_sgf_directory(train_sgf_directory)
    dataset_id = f"{source}_g{max_games}_s{seed}_bs{board_size}_p{num_planes}"
    if test_mode:
        return f"test_{dataset_id}"
    return dataset_id


def dataset_directory(dataset_id: str) -> Path:
    return DATASETS_ROOT / dataset_id


def manifest_path(data_directory: str | Path) -> Path:
    return Path(data_directory) / MANIFEST_FILENAME


def load_manifest(data_directory: str | Path) -> Optional[DatasetManifest]:
    path = manifest_path(data_directory)
    if not path.exists():
        return None
    return DatasetManifest.load(path)


def ensure_manifest(
    data_directory: str | Path,
    *,
    dataset_id: str,
    board_size: int,
    num_planes: int,
) -> DatasetManifest:
    data_dir = Path(data_directory)
    existing = load_manifest(data_dir)
    if existing is not None:
        return existing
    return DatasetManifest(
        dataset_id=dataset_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        board_size=board_size,
        num_planes=num_planes,
    )


def record_split(
    data_directory: str | Path,
    *,
    dataset_id: str,
    board_size: int,
    num_planes: int,
    split: SplitRecord,
) -> DatasetManifest:
    manifest = ensure_manifest(
        data_directory,
        dataset_id=dataset_id,
        board_size=board_size,
        num_planes=num_planes,
    )
    manifest.splits[split.prefix] = split
    manifest.save(manifest_path(data_directory))
    return manifest


def cache_is_valid(
    data_directory: str | Path,
    *,
    prefix: str,
    sgf_directory: str | Path,
    max_games: Optional[int],
    seed: int,
    board_size: int,
    num_planes: int,
) -> bool:
    data_dir = Path(data_directory)
    features_path = data_dir / f"{prefix}_features.npy"
    labels_path = data_dir / f"{prefix}_labels.npy"
    if not features_path.exists() or not labels_path.exists():
        return False

    manifest = load_manifest(data_dir)
    if manifest is None:
        return False

    split = manifest.split_record(prefix)
    if split is None:
        return False
    return split.matches_request(
        sgf_directory=sgf_directory,
        max_games=max_games,
        seed=seed,
        board_size=board_size,
        num_planes=num_planes,
    )


def record_agent_usage(
    *,
    agent_id: str,
    dataset_id: str,
    dataset_dir: str | Path,
    usage_info: Dict[str, Any],
) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    entry = {
        "timestamp": timestamp,
        "agent_id": agent_id,
        "dataset_id": dataset_id,
        "dataset_dir": str(Path(dataset_dir).resolve()),
        **usage_info,
    }

    USAGE_ROOT.mkdir(parents=True, exist_ok=True)
    usage_log = USAGE_ROOT / f"{agent_id}.jsonl"
    with usage_log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")

    manifest = load_manifest(dataset_dir)
    if manifest is not None:
        manifest.used_by.append(entry)
        manifest.save(manifest_path(dataset_dir))
