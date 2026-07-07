"""Download and unpack the MovieLens 1M dataset into data/ml-1m/.

Usage: python data/download.py
Idempotent: skips the download if the extracted files already exist.
"""

import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DATA_DIR, RAW_DIR

URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def main() -> None:
    if (RAW_DIR / "ratings.dat").exists():
        print(f"Dataset already present at {RAW_DIR}, skipping download.")
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "ml-1m.zip"
    print(f"Downloading {URL} ...")
    urllib.request.urlretrieve(URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DATA_DIR)
    print(f"Extracted to {RAW_DIR}")


if __name__ == "__main__":
    main()
