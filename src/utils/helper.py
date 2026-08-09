from collections.abc import Iterable
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


def ensure_directories_exist(paths: Iterable[PathLike]) -> None:
    """
    Create directories if they do not already exist.
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)