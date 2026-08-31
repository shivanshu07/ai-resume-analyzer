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


def pytest_configure(config):
    """
    Silences one specific, known-harmless warning that comes
    from INSIDE the deepeval library itself (deepeval/utils.py
    calling the deprecated asyncio.get_event_loop() pattern),
    not from any code in this repo. Confirmed as of deepeval
    4.2.0 (the latest release available) that no upgrade fixes
    this -- it's current, unresolved upstream behavior.

    Scoped narrowly to this exact message + category + module
    so it does NOT hide unrelated DeprecationWarnings that
    might actually matter, from deepeval or anywhere else.
    """

    config.addinivalue_line(
        "filterwarnings",
        "ignore:There is no current event loop:DeprecationWarning:deepeval.utils"
    )