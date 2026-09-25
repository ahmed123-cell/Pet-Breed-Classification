"""
Sample generator creating realistic pet sample images across all 37 classes
for rapid local prototyping, CI verification, and offline benchmarking.
"""

import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

from pet_breed.config import ANNOTATIONS_DIR, IMAGES_DIR, get_class_names, get_species_map


def generate_sample_dataset(samples_per_class: int = 5) -> None:
    """
    Generate synthetic sample images and annotation files for all 37 classes.
    """
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    classes = get_class_names()
    species_map = get_species_map()

    trainval_lines = []
    test_lines = []

    print(f"Generating {samples_per_class} sample images per class for {len(classes)} classes...")
    random.seed(42)

    for class_idx, class_name in enumerate(classes):
        species = species_map.get(class_name, "cat")
        # Format filename stem according to Oxford convention
        if species == "cat":
            prefix = class_name.replace(" ", "_")
        else:
            prefix = class_name.lower().replace(" ", "_")

        # Color palette for class visual variety
        base_color = (
            random.randint(50, 200),
            random.randint(50, 200),
            random.randint(50, 200),
        )

        for i in range(1, samples_per_class + 1):
            stem = f"{prefix}_{i}"
            img_path = IMAGES_DIR / f"{stem}.jpg"

            # Create image with textured pattern
            img = Image.new("RGB", (256, 256), color=base_color)
            draw = ImageDraw.Draw(img)
            draw.ellipse(
                [40 + i * 2, 40 + i * 2, 216 - i * 2, 216 - i * 2],
                fill=(
                    (base_color[0] + 40) % 255,
                    (base_color[1] + 30) % 255,
                    (base_color[2] + 50) % 255,
                ),
            )
            draw.text((50, 120), f"{class_name} #{i}", fill=(255, 255, 255))
            img.save(img_path, "JPEG", quality=95)

            # Partition into trainval vs test
            species_id = 1 if species == "cat" else 2
            line = f"{stem} {class_idx + 1} {species_id} 1"
            if i % 2 == 1:
                trainval_lines.append(line)
            else:
                test_lines.append(line)

    with open(ANNOTATIONS_DIR / "trainval.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(trainval_lines) + "\n")

    with open(ANNOTATIONS_DIR / "test.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(test_lines) + "\n")

    print(f"Sample dataset generated: {len(list(IMAGES_DIR.glob('*.jpg')))} images in {IMAGES_DIR}")


if __name__ == "__main__":
    generate_sample_dataset()
