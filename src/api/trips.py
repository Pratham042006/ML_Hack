"""Finding trips on disk. Only ever looks at the folder names and the .track/.nodes/.arcs files.

It never opens a .route file, so reference data cannot leak through this module.
"""
from pathlib import Path

REQUIRED_SUFFIXES = (".track", ".nodes", ".arcs")


class DataDirError(Exception):
    """The configured dataset directory is missing or unreadable."""


def list_trip_ids(data_dir: Path) -> list:
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise DataDirError(
            f"Dataset directory '{data_dir}' does not exist. "
            "Set PATHSNAP_DATA_DIR to a folder containing one sub-folder per trip."
        )
    ids = []
    for child in sorted(data_dir.iterdir()):
        if child.is_dir() and all((child / f"{child.name}{s}").is_file() for s in REQUIRED_SUFFIXES):
            ids.append(child.name)
    return ids


def trip_path(data_dir: Path, trip_id: str, available: list) -> Path:
    """Resolve a trip folder, but only for IDs that appeared in the listing (blocks path tricks)."""
    if trip_id not in available:
        raise KeyError(trip_id)
    return Path(data_dir) / trip_id
