from pathlib import Path


def ensure_directories_exist(paths):
    """
    Create directories if they do not already exist.
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)