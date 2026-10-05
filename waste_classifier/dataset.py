"""Fetch the Kaggle "Waste Classification data" set (techsash/waste-classification-data)."""
from pathlib import Path

KAGGLE_DATASET = "techsash/waste-classification-data"


def find_split_root(path: Path) -> Path:
    """Return the shallowest directory under `path` that contains TRAIN/ and TEST/.

    The Kaggle archive ships the data twice (DATASET/ and DATASET/DATASET/), so take the first.
    """
    path = Path(path)
    for candidate in sorted([path, *path.rglob("*")], key=lambda p: len(p.parts)):
        if (candidate / "TRAIN").is_dir() and (candidate / "TEST").is_dir():
            return candidate
    raise FileNotFoundError(f"No directory with TRAIN/ and TEST/ found under {path}")


def download_dataset() -> Path:
    """Download (or reuse the cached copy of) the dataset and return its TRAIN/TEST root."""
    import kagglehub

    return find_split_root(Path(kagglehub.dataset_download(KAGGLE_DATASET)))
