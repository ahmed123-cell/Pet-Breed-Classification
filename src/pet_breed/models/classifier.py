"""
Production PetBreedClassifier wrapping 37-class PyTorch backbone, canonical eval transform,
temperature calibration, and selective abstention gating.
"""

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

from pet_breed.config import (
    DEFAULT_ABSTENTION_THRESHOLD,
    DEFAULT_NUM_CLASSES,
    MODEL_VERSION,
    get_class_names,
    get_species_map,
)
from pet_breed.models.backbones import get_model
from pet_breed.models.transforms import CanonicalPreprocessor


class PetBreedClassifier:
    """
    Production-ready classifier service wrapper.
    Loads trained 37-class PetClassifierBackbone weights, applies temperature-calibrated
    softmax probabilities, and enforces abstention gating on low-confidence inputs.
    """

    def __init__(
        self,
        backbone_name: str = "resnet50",
        num_classes: int = DEFAULT_NUM_CLASSES,
        weights_path: str | Path | None = None,
        temperature: float = 1.0,
        abstention_threshold: float = DEFAULT_ABSTENTION_THRESHOLD,
        device: str | None = None,
        model_version: str = MODEL_VERSION,
    ):
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.temperature = float(temperature)
        self.abstention_threshold = float(abstention_threshold)
        self.model_version = model_version

        # Canonical sorted class names & species map
        self.class_names: list[str] = get_class_names()
        self.species_map: dict[str, str] = get_species_map()

        state_dict = None
        if weights_path is not None:
            p = Path(weights_path)
            if p.exists():
                checkpoint = torch.load(p, map_location=self.device)
                if isinstance(checkpoint, dict):
                    if "backbone" in checkpoint:
                        self.backbone_name = checkpoint["backbone"]
                    if "num_classes" in checkpoint:
                        self.num_classes = checkpoint["num_classes"]
                    if (
                        "temperature" in checkpoint
                        and checkpoint["temperature"] is not None
                    ):
                        self.temperature = float(checkpoint["temperature"])
                    if (
                        "abstention_threshold" in checkpoint
                        and checkpoint["abstention_threshold"] is not None
                    ):
                        self.abstention_threshold = float(
                            checkpoint["abstention_threshold"]
                        )
                    if (
                        "model_version" in checkpoint
                        and checkpoint["model_version"] is not None
                    ):
                        self.model_version = checkpoint["model_version"]
                    if "class_names" in checkpoint and checkpoint["class_names"]:
                        self.class_names = list(checkpoint["class_names"])
                    if "species_map" in checkpoint and checkpoint["species_map"]:
                        self.species_map = dict(checkpoint["species_map"])

                    if "state_dict" in checkpoint:
                        state_dict = checkpoint["state_dict"]
                    elif "model_state_dict" in checkpoint:
                        state_dict = checkpoint["model_state_dict"]
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint

        # Construct trained 37-class architecture
        self.model = get_model(
            backbone_name=self.backbone_name,
            num_classes=self.num_classes,
            pretrained=False,
        )

        if state_dict is not None:
            self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

        # Canonical Preprocessor loaded once at startup
        self.preprocessor = CanonicalPreprocessor()

    @torch.inference_mode()
    def predict(
        self,
        image: Image.Image,
        return_embeddings: bool = False,
    ) -> dict[str, Any]:
        """
        Run inference on a single PIL image (accepting RGB, RGBA, CMYK, Grayscale).
        Outputs calibrated confidence scores with abstention gating.
        """
        tensor = self.preprocessor.preprocess_pil(image).unsqueeze(0).to(self.device)

        # 1. Forward pass returning raw 37-class logits
        logits = self.model(tensor)[0]

        # 2. Temperature scaling calibration
        calibrated_logits = logits / max(self.temperature, 1e-6)
        probs = F.softmax(calibrated_logits, dim=-1)

        # 3. Top-1 and Top-3 Breeds from the 37 committed classes
        top_probs, top_indices = torch.topk(probs, k=min(3, self.num_classes))

        top1_idx = top_indices[0].item()
        top1_prob = top_probs[0].item()
        top1_breed = self.class_names[top1_idx]
        detected_species = self.species_map.get(top1_breed, "unknown")

        top_3 = []
        for p, idx in zip(top_probs.tolist(), top_indices.tolist(), strict=False):
            b_name = self.class_names[idx]
            top_3.append(
                {
                    "breed": b_name,
                    "species": self.species_map.get(b_name, "unknown"),
                    "confidence": round(float(p), 4),
                }
            )

        decision = (
            "confident" if top1_prob >= self.abstention_threshold else "uncertain"
        )

        result = {
            "breed": top1_breed,
            "species": detected_species,
            "confidence": round(float(top1_prob), 4),
            "top_3": top_3,
            "decision": decision,
            "model_version": self.model_version,
        }

        if return_embeddings:
            features = self.model.extract_features(tensor)
            result["embedding"] = features.squeeze(0).cpu().numpy()

        return result
