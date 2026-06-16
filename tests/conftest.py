"""Pytest configuration."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DLGO_ROOT = ROOT / "DLGO - code"
if DLGO_ROOT.exists() and str(DLGO_ROOT) not in sys.path:
    sys.path.insert(0, str(DLGO_ROOT))
