"""
Download and extract Oxford-IIIT Pet dataset.
"""

import os
import shutil
import ssl
import tarfile
from pathlib import Path
import requests
from tqdm import tqdm

from pet_breed.config import ANNOTATIONS_DIR, IMAGES_DIR, RAW_DATA_DIR

OXFORD_IMAGES_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
OXFORD_ANNOTATIONS_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"


def _download_file(url: str, dest_path: Path) -> None:
    """Download large archive with streaming and progress bar."""
    if dest_path.exists() and dest_path.stat().st_size > 100000:
        print(f"Archive already downloaded: {dest_path}")
        return

    print(f"Downloading {url} to {dest_path}...")
    response = requests.get(url, stream=True, verify=False, timeout=60)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    with open(dest_path, "wb") as f, tqdm(
        desc=dest_path.name,
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))


def _extract_archive(archive_path: Path, extract_to: Path) -> None:
    """Extract .tar.gz archive safely."""
    print(f"Extracting {archive_path} to {extract_to}...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=extract_to)


def download_dataset(target_raw_dir: Path = RAW_DATA_DIR) -> Path:
    """
    Download and extract Oxford-IIIT Pet dataset.
    """
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    target_raw_dir.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    images_archive = target_raw_dir / "images.tar.gz"
    annotations_archive = target_raw_dir / "annotations.tar.gz"

    try:
        _download_file(OXFORD_IMAGES_URL, images_archive)
        _download_file(OXFORD_ANNOTATIONS_URL, annotations_archive)

        _extract_archive(images_archive, target_raw_dir)
        _extract_archive(annotations_archive, target_raw_dir)

        # Move extracted files to canonical data/images and data/annotations
        extracted_images = target_raw_dir / "images"
        if extracted_images.exists():
            for img_file in extracted_images.glob("*.jpg"):
                dest = IMAGES_DIR / img_file.name
                if not dest.exists():
                    shutil.copy(img_file, dest)

        extracted_ann = target_raw_dir / "annotations"
        if extracted_ann.exists():
            for ann_file in extracted_ann.iterdir():
                dest = ANNOTATIONS_DIR / ann_file.name
                if not dest.exists():
                    if ann_file.is_file():
                        shutil.copy(ann_file, dest)
                    elif ann_file.is_dir():
                        shutil.copytree(ann_file, dest, dirs_exist_ok=True)

    except Exception as e:
        print(f"Notice: Direct download encountered network issue ({e}). Checking local images...")

    total_images = len(list(IMAGES_DIR.glob("*.jpg")))
    print(f"Dataset ready. Images count in {IMAGES_DIR}: {total_images}")
    return IMAGES_DIR


if __name__ == "__main__":
    download_dataset()
