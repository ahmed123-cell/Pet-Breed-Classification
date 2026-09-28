"""
Fast, Reliable Downloader and Extractor for the Oxford-IIIT Pet Dataset.
Uses the official Thor CDN mirror with strict size and archive validation.
"""

import os
import shutil
import tarfile
import time
from pathlib import Path
import requests
from tqdm import tqdm
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from pet_breed.config import ANNOTATIONS_DIR, IMAGES_DIR, RAW_DATA_DIR

THOR_IMAGES_URL = "https://thor.robots.ox.ac.uk/datasets/pets/images.tar.gz"
THOR_ANNOTATIONS_URL = "https://thor.robots.ox.ac.uk/datasets/pets/annotations.tar.gz"

MIN_IMAGES_BYTES = 790_000_000      # ~755 MB
MIN_ANNOTATIONS_BYTES = 19_000_000  # ~18.2 MB


def _download_stream(url: str, dest_path: Path, min_bytes: int, max_retries: int = 15) -> None:
    """Download archive reliably with streaming and retry."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if existing archive is full size
    if dest_path.exists():
        if dest_path.stat().st_size >= min_bytes:
            print(f"Verified complete archive exists: {dest_path.name} ({dest_path.stat().st_size / 1e6:.1f} MB)")
            return
        else:
            print(f"Removing incomplete archive ({dest_path.stat().st_size / 1e6:.1f} MB < {min_bytes / 1e6:.1f} MB): {dest_path.name}")
            dest_path.unlink()

    temp_path = dest_path.with_suffix(".part")
    if temp_path.exists():
        temp_path.unlink()

    print(f"\nConnecting to {url}...")
    
    for attempt in range(1, max_retries + 1):
        try:
            with requests.get(url, stream=True, verify=False, timeout=120) as r:
                r.raise_for_status()
                total_size = int(r.headers.get("content-length", 0))
                print(f"Downloading {dest_path.name} ({total_size / 1e6:.1f} MB)...")
                
                with open(temp_path, "wb") as f, tqdm(
                    desc=dest_path.name,
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))
                            
            if temp_path.exists() and temp_path.stat().st_size >= min_bytes:
                temp_path.rename(dest_path)
                print(f"Successfully downloaded and verified {dest_path.name} ({dest_path.stat().st_size / 1e6:.1f} MB)!")
                return
            else:
                print(f"Download incomplete. Retrying...")
                if temp_path.exists():
                    temp_path.unlink()
        except Exception as e:
            print(f"Download attempt {attempt}/{max_retries} failed: {e}. Retrying in 5s...")
            time.sleep(5)
            
    raise RuntimeError(f"Failed to download {url} after {max_retries} attempts.")


def _extract_archive(archive_path: Path, extract_to: Path) -> None:
    """Extract .tar.gz archive safely."""
    print(f"Extracting {archive_path.name} to {extract_to}...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=extract_to)


def download_dataset(target_raw_dir: Path = RAW_DATA_DIR) -> Path:
    """Download and unpack authentic 7,349 photos."""
    target_raw_dir.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    images_archive = target_raw_dir / "images.tar.gz"
    annotations_archive = target_raw_dir / "annotations.tar.gz"

    # 1. Download full archives
    _download_stream(THOR_ANNOTATIONS_URL, annotations_archive, min_bytes=MIN_ANNOTATIONS_BYTES)
    _download_stream(THOR_IMAGES_URL, images_archive, min_bytes=MIN_IMAGES_BYTES)

    # 2. Extract
    _extract_archive(annotations_archive, target_raw_dir)
    _extract_archive(images_archive, target_raw_dir)

    # 3. Sync extracted images into data/images
    extracted_images = target_raw_dir / "images"
    if extracted_images.exists():
        print(f"Syncing real photographs into {IMAGES_DIR}...")
        img_files = list(extracted_images.glob("*.jpg"))
        for img_file in tqdm(img_files, desc="Copying real images"):
            dest = IMAGES_DIR / img_file.name
            shutil.copy2(img_file, dest)

    extracted_ann = target_raw_dir / "annotations"
    if extracted_ann.exists():
        for ann_file in extracted_ann.iterdir():
            dest = ANNOTATIONS_DIR / ann_file.name
            if ann_file.is_file():
                shutil.copy2(ann_file, dest)
            elif ann_file.is_dir():
                shutil.copytree(ann_file, dest, dirs_exist_ok=True)

    total_images = len(list(IMAGES_DIR.glob("*.jpg")))
    print(f"\n" + "=" * 50)
    print(f"Success! Real Pet Photos in {IMAGES_DIR}: {total_images}")
    print("=" * 50)
    return IMAGES_DIR


if __name__ == "__main__":
    download_dataset()
