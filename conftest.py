"""
Ensures the repository root is on sys.path so that
`from src...` imports resolve correctly no matter how pytest
is invoked -- bare `pytest`, `python -m pytest`, from an IDE's
test runner, or from CI.

Without this, running pytest as a plain command (not `python
-m pytest`) does NOT add the current working directory to
sys.path, which causes every test file that imports from the
src/ package to fail collection with:

    ModuleNotFoundError: No module named 'src'

This file makes that failure mode impossible regardless of
invocation style, which is safer long-term than relying on
always remembering to type `python -m pytest`.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))