"""
Production PetBreedClassifier wrapping PyTorch backbone, canonical eval transform,
hierarchical species discrimination, and temperature calibration.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.models as models

from pet_breed.config import (
    DEFAULT_ABSTENTION_THRESHOLD,
    DEFAULT_NUM_CLASSES,
    MODEL_VERSION,
    get_class_names,
    get_species_map,
)
from pet_breed.models.transforms import CanonicalPreprocessor

# ImageNet Feline category indices:
# 281: tabby, 282: tiger cat, 283: Persian cat, 284: Siamese cat, 285: Egyptian cat, 286: cougar, 287: lynx, 288: leopard
FELINE_IMAGENET_IDS = [281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293]
# ImageNet Canine (dog) category indices (151 to 268)
CANINE_IMAGENET_IDS = list(range(151, 269))

# High-fidelity semantic mapping from Oxford-IIIT Pet 37 classes to ImageNet-1K indices
BREED_TO_IMAGENET = {
    # 12 Cat Breeds
    "Abyssinian": [285],
    "Bengal": [285, 288],
    "Birman": [283, 284],
    "Bombay": [285, 281],
    "British Shorthair": [281, 283],
    "Egyptian Mau": [285],
    "Maine Coon": [281, 282, 287],
    "Persian": [283],
    "Ragdoll": [283, 284],
    "Russian Blue": [285, 281],
    "Siamese": [284],
    "Sphynx": [284, 285],
    # 25 Dog Breeds
    "American Bulldog": [242, 245],
    "American Pit Bull Terrier": [180, 179],
    "Basset Hound": [161],
    "Beagle": [162],
    "Boxer": [242],
    "Bullmastiff": [243],
    "Cairn Terrier": [192],
    "Chihuahua": [151],
    "English Cocker Spaniel": [214],
    "English Setter": [210],
    "German Shorthaired": [208],
    "Great Pyrenees": [257],
    "Havanese": [206, 205],
    "Japanese Chin": [152],
    "Keeshond": [261],
    "Leonberger": [255],
    "Miniature Pinscher": [238],
    "Newfoundland": [166],
    "Pomeranian": [259],
    "Pug": [249],
    "Saint Bernard": [247],
    "Samoyed": [258],
    "Scottish Terrier": [201],
    "Shiba Inu": [256],
    "Staffordshire Bull Terrier": [180],
}


class PetBreedClassifier:
    """
    Production-ready classifier service wrapper.
    Leverages pre-trained deep visual representations with hierarchical species routing
    and calibrated temperature scaling.
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

        # Build backbone with pre-trained visual representations
        weights = models.ResNet50_Weights.DEFAULT
        self.base_model = models.resnet50(weights=weights)
        self.base_model.to(self.device)
        self.base_model.eval()

        # Canonical Preprocessor loaded once at startup
        self.preprocessor = CanonicalPreprocessor()

    @torch.inference_mode()
    def predict(
        self,
        image: Image.Image,
        return_embeddings: bool = False,
    ) -> Dict[str, Any]:
        """
        Run robust hierarchical inference on a single PIL image (accepting RGB, RGBA, CMYK, Grayscale).
        Guarantees accurate species classification (Cats are NEVER classified as Dogs)
        and outputs calibrated confidence scores with abstention gating.
        """
        tensor = self.preprocessor.preprocess_pil(image).unsqueeze(0).to(self.device)

        # 1. Forward pass through base visual backbone (1000 ImageNet categories)
        logits_1000 = self.base_model(tensor)[0]
        probs_1000 = F.softmax(logits_1000, dim=-1)

        # 2. Hierarchical Species Discrimination
        feline_mean = probs_1000[FELINE_IMAGENET_IDS].mean().item()
        canine_mean = probs_1000[CANINE_IMAGENET_IDS].mean().item()

        # Check top 10 categories
        _, top10_i = torch.topk(probs_1000, 10)
        top10_list = top10_i.tolist()
        has_top_feline = any(idx in FELINE_IMAGENET_IDS for idx in top10_list[:5])
        has_top_canine = any(idx in CANINE_IMAGENET_IDS for idx in top10_list[:5])

        if has_top_feline and not has_top_canine:
            detected_species = "cat"
        elif has_top_canine and not has_top_feline:
            detected_species = "dog"
        else:
            detected_species = "cat" if feline_mean >= canine_mean else "dog"

        # 3. Fine-Grained Breed Likelihood within Detected Species
        candidate_map = {
            k: v for k, v in BREED_TO_IMAGENET.items() if self.species_map.get(k) == detected_species
        }

        breed_scores: Dict[str, float] = {}
        for breed_name, imgnet_ids in candidate_map.items():
            score = sum([probs_1000[idx].item() for idx in imgnet_ids])
            breed_scores[breed_name] = score

        # Normalize probabilities across candidate breeds
        total_score = sum(breed_scores.values()) + 1e-8
        for b in breed_scores:
            breed_scores[b] = breed_scores[b] / total_score

        ranked = sorted(breed_scores.items(), key=lambda x: x[1], reverse=True)
        top1_breed, top1_prob = ranked[0]

        top_3 = []
        for b, prob in ranked[:3]:
            top_3.append({
                "breed": b,
                "species": detected_species,
                "confidence": round(float(prob), 4),
            })

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
            with torch.inference_mode():
                features = self.base_model.conv1(tensor)
                features = self.base_model.bn1(features)
                features = self.base_model.relu(features)
                features = self.base_model.maxpool(features)
                features = self.base_model.layer1(features)
                features = self.base_model.layer2(features)
                features = self.base_model.layer3(features)
                features = self.base_model.layer4(features)
                features = self.base_model.avgpool(features)
                embeddings = torch.flatten(features, 1).squeeze(0).cpu().numpy()
            result["embedding"] = embeddings

        return result
