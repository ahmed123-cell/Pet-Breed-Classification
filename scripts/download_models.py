"""
Download and verify production model checkpoints.

Fetches the trained 37-class Oxford-IIIT Pet breed classification models
from the official GitHub release and verifies file integrity via SHA256 checksums.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

import torch

from pet_breed.config import (
    CHECKPOINTS_DIR,
    DEFAULT_NUM_CLASSES,
    EXPORTED_MODELS_DIR,
)

# Official GitHub release assets
RELEASE_TAG = "v1.0.0"
BASE_URL = f"https://github.com/ahmed123-cell/Pet-Breed-Classification/releases/download/{RELEASE_TAG}"

MODELS_MANIFEST = {
    "resnet50_calibrated.pth": {
        "target_dir": EXPORTED_MODELS_DIR,
        "url": f"{BASE_URL}/resnet50_calibrated.pth",
        "sha256": "d02a755d3e92025e9969b2c08176afb1bc183a783bbf773238f7cf51363dabca",
        "size_bytes": 94657479,
        "description": "Production calibrated 37-class ResNet-50 (temperature=0.2779, threshold=0.65, top1=91.4%)",
    },
    "resnet50_best.pth": {
        "target_dir": CHECKPOINTS_DIR,
        "url": f"{BASE_URL}/resnet50_best.pth",
        "sha256": "3258b2b5eeea44a5696eda73bd37945d579be236527bfb326cf5817fe64c90f2",
        "size_bytes": 94655203,
        "description": "Best training checkpoint (epoch 5, 37 classes)",
    },
}


def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 hex digest of a file in streaming chunks."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_checkpoint(file_path: Path, expected_sha256: str | None = None) -> bool:
    """Verify that a checkpoint exists, matches checksum, and loads cleanly in PyTorch."""
    if not file_path.exists():
        return False

    if expected_sha256:
        file_sha = compute_sha256(file_path)
        if file_sha.lower() != expected_sha256.lower():
            print(
                f"[WARN] Checksum mismatch for {file_path.name}:\n"
                f"  Expected: {expected_sha256}\n"
                f"  Got:      {file_sha}"
            )
            return False

    try:
        ckpt = torch.load(file_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt
        if not isinstance(state_dict, dict):
            return False
        # Verify 37 classes in output head
        head_w = state_dict.get("head.weight")
        if head_w is not None and head_w.shape[0] != DEFAULT_NUM_CLASSES:
            return False
        return True
    except Exception as e:
        print(f"[WARN] Failed to load checkpoint {file_path}: {e}")
        return False


def download_file(url: str, dest_path: Path, expected_size: int | None = None) -> None:
    """Download a file with progress reporting."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(".tmp")

    print(f"Downloading from {url}...")
    headers = {"User-Agent": "PetBreedModelDownloader/1.0"}
    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req) as response, open(tmp_path, "wb") as out_file:
            total_size = int(response.headers.get("Content-Length", expected_size or 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1 MB chunks

            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    mb_down = downloaded / (1024 * 1024)
                    mb_tot = total_size / (1024 * 1024)
                    print(
                        f"\r  Progress: {pct:5.1f}% ({mb_down:6.1f} / {mb_tot:6.1f} MB)",
                        end="",
                        flush=True,
                    )
                else:
                    mb_down = downloaded / (1024 * 1024)
                    print(f"\r  Downloaded {mb_down:6.1f} MB", end="", flush=True)

        print()  # newline after progress bar
        tmp_path.replace(dest_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def download_models(force: bool = False) -> dict[str, Path]:
    """
    Ensure all official model checkpoints are downloaded and verified.

    Returns dict mapping artifact filename to verified Path.
    """
    results: dict[str, Path] = {}

    for filename, info in MODELS_MANIFEST.items():
        target_path = info["target_dir"] / filename
        expected_sha = info["sha256"]

        if not force and verify_checkpoint(target_path, expected_sha):
            print(
                f"[OK] {filename} is already present and verified ({info['description']})."
            )
            results[filename] = target_path
            continue

        print(f"\n[INFO] Fetching {filename}: {info['description']}")
        try:
            download_file(info["url"], target_path, expected_size=info["size_bytes"])
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Failed to download {filename} from {info['url']}: {e}.\n"
                "Please check your internet connection or download manually into:\n"
                f"  {target_path}"
            ) from e

        if not verify_checkpoint(target_path, expected_sha):
            if target_path.exists():
                target_path.unlink()
            raise RuntimeError(
                f"Integrity check failed for downloaded file {target_path}. Removed file."
            )

        print(f"[SUCCESS] Verified {filename} (SHA256: {expected_sha[:12]}...)")
        results[filename] = target_path

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and verify official Pet Breed Classification model checkpoints."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and overwrite existing checkpoints.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Pet Breed Classification - Model Artifact Downloader")
    print(f"Target Release: {RELEASE_TAG}")
    print("=" * 70)

    try:
        downloaded = download_models(force=args.force)
        print("\nAll model checkpoints are ready:")
        for _name, path in downloaded.items():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  - {path} ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
