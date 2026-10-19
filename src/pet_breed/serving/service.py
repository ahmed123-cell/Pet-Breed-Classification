"""
BentoML Service definition with micro-batching runner.
"""

from typing import Any

import bentoml
from bentoml.io import JSON
from bentoml.io import Image as BentoImage
from PIL import Image

from pet_breed.config import (
    DEFAULT_ABSTENTION_THRESHOLD,
    DEFAULT_NUM_CLASSES,
    MODEL_VERSION,
)
from pet_breed.models.classifier import PetBreedClassifier

# Global classifier instance
classifier = PetBreedClassifier(
    backbone_name="resnet50",
    num_classes=DEFAULT_NUM_CLASSES,
    abstention_threshold=DEFAULT_ABSTENTION_THRESHOLD,
    model_version=MODEL_VERSION,
)


@bentoml.service(
    name="pet_breed_classifier_service",
    resources={"cpu": "2"},
)
class PetBreedBentoService:
    def __init__(self):
        self.classifier = classifier

    @bentoml.api(batchable=True, batch_dim=0)
    def predict(self, images: list[Image.Image]) -> list[dict[str, Any]]:
        """
        Batchable inference endpoint supporting dynamic micro-batching.
        """
        results = []
        for img in images:
            res = self.classifier.predict(img)
            results.append(res)
        return results


# Fallback standard bentoml.Service for BentoML 1.0+ legacy interface compatibility
svc = bentoml.Service("pet_breed_service")


@svc.api(input=BentoImage(), output=JSON())
def predict(image: Image.Image) -> dict[str, Any]:
    return classifier.predict(image)
