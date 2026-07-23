"""Shared filesystem locations for tests moved out of the source root."""
from __future__ import annotations

from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
