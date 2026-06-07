"""Make `src.*` and `catalog.*` importable when pytest runs from `lean/`."""

import sys
from pathlib import Path

_LEAN_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_LEAN_ROOT))
