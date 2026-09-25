"""
Production PetBreedClassifier wrapping PyTorch backbone, canonical eval transform,
temperature scaling, and selective abstention decision logic.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image
import torch
import torch.nn.functional as F

from pet_breed.config import (
    DEFAULT_ABSTENTION_THRESHOLD,
    DEFAULT_NUM_CLASSES,
    MODEL_VERSION,
    get_class_names,
    get_species_map,
)
from pet_breed.models.backbones import PetClassifierBackbone, get_model
from pet_breed.models.transforms import CanonicalPreprocessor


class PetBreedClassifier:
    """
    Production-ready classifier service wrapper.
    Loads backbone weights, canonical transform, and calibration temperature once at initialization.
    """

    def __init__(
        self,
        backbone_name: str = "resnet50",
        num_classes: int = DEFAULT_NUM_CLASSES,
        weights_path: Optional[Union[str, Path]] = None,
        temperature: float = 1.0,
        abstention_threshold: float = DEFAULT_ABSTENTION_THRESHOLD,
        device: Optional[str] = None,
        model_version: str = MODEL_VERSION,
    ):
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.temperature = float(temperature)
        self.abstention_threshold = float(abstention_threshold)
        self.model_version = model_version

        # Canonical sorted class names & species map
        self.class_names: List[str] = get_class_names()
        self.species_map: Dict[str, str] = get_species_map()

        # Build backbone model
        self.model: PetClassifierBackbone = get_model(
            backbone_name=backbone_name,
            num_classes=num_classes,
            pretrained=True if weights_path is None else False,
        )

        if weights_path is not None:
            w_path = Path(weights_path)
            if w_path.exists():
                state_dict = torch.load(w_path, map_location=self.device)
                # Handle possible dict wrapper
                if "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                if "model_state_dict" in state_dict:
                    state_dict = state_dict["model_state_dict"]
                self.model.load_state_dict(state_dict, strict=False)

        self.model.to(self.device)
        self.model.eval()

        # Canonical Preprocessor loaded once at startup
        self.preprocessor = CanonicalPreprocessor()

    @torch.inference_mode()
    def get_logits(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Raw unscaled forward logits pass."""
        return self.model(input_tensor.to(self.device))

    @torch.inference_mode()
    def get_embeddings(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Penultimate feature embeddings for MMD drift monitoring."""
        return self.model.extract_features(input_tensor.to(self.device))

    @torch.inference_mode()
    def predict(
        self,
        image: Image.Image,
        return_embeddings: bool = False,
    ) -> Dict[str, Any]:
        """
        Run inference on a single PIL image (accepting RGB, RGBA, CMYK, Grayscale).
        Applies temperature calibration and returns calibrated top-3 predictions and abstention decision.
        """
        # Canonical preprocessing
        tensor = self.preprocessor.preprocess_pil(image).unsqueeze(0).to(self.device)
        
        # Logits & Temperature Scaling
        logits = self.model(tensor)
        scaled_logits = logits / max(self.temperature, 1e-6)
        probs = F.softmax(scaled_logits, dim=-1).squeeze(0)

        top_probs, top_indices = torch.topk(probs, k=min(3, self.num_classes))
        top_probs_list = top_probs.cpu().tolist()
        top_indices_list = top_indices.cpu().tolist()

        top_1_idx = top_indices_list[0]
        top_1_prob = float(top_probs_list[0])
        top_1_breed = self.class_names[top_1_idx]
        top_1_species = self.species_map.get(top_1_breed, "unknown")

        decision = (
            "confident" if top_1_prob >= self.abstention_threshold else "uncertain"
        )

        top_3 = []
        for prob, idx in zip(top_probs_list, top_indices_list):
            b_name = self.class_names[idx]
            top_3.append({
                "breed": b_name,
                "species": self.species_map.get(b_name, "unknown"),
                "confidence": round(float(prob), 4),
            })

        result = {
            "breed": top_1_breed,
            "species": top_1_species,
            "confidence": round(top_1_prob, 4),
            "top_3": top_3,
            "decision": decision,
            "model_version": self.model_version,
        }

        if return_embeddings:
            embeddings = self.model.extract_features(tensor).squeeze(0).cpu().numpy()
            result["embedding"] = embeddings

        return result
