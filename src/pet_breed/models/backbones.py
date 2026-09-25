"""
Model backbone factories for ResNet-50, ResNet-18, and MobileNetV3-Small.
"""

from typing import Tuple, Type
import torch
import torch.nn as nn
import torchvision.models as models

from pet_breed.config import DEFAULT_NUM_CLASSES


class PetClassifierBackbone(nn.Module):
    """
    Backbone wrapper exposing logits and penultimate feature embeddings.
    """

    def __init__(
        self,
        backbone_name: str = "resnet50",
        num_classes: int = DEFAULT_NUM_CLASSES,
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        self.num_classes = num_classes

        if self.backbone_name == "resnet50":
            try:
                import ssl
                ssl._create_default_https_context = ssl._create_unverified_context
            except Exception:
                pass
            try:
                weights = models.ResNet50_Weights.DEFAULT if pretrained else None
                base_model = models.resnet50(weights=weights)
            except Exception:
                base_model = models.resnet50(weights=None)
            in_features = base_model.fc.in_features
            base_model.fc = nn.Identity()
            self.backbone = base_model
            self.head = nn.Linear(in_features, num_classes)
            self.embedding_dim = in_features

        elif self.backbone_name == "resnet18":
            try:
                import ssl
                ssl._create_default_https_context = ssl._create_unverified_context
            except Exception:
                pass
            try:
                weights = models.ResNet18_Weights.DEFAULT if pretrained else None
                base_model = models.resnet18(weights=weights)
            except Exception:
                base_model = models.resnet18(weights=None)
            in_features = base_model.fc.in_features
            base_model.fc = nn.Identity()
            self.backbone = base_model
            self.head = nn.Linear(in_features, num_classes)
            self.embedding_dim = in_features

        elif self.backbone_name in ["mobilenet_v3_small", "mobilenetv3_small"]:
            try:
                import ssl
                ssl._create_default_https_context = ssl._create_unverified_context
            except Exception:
                pass
            try:
                weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
                base_model = models.mobilenet_v3_small(weights=weights)
            except Exception:
                base_model = models.mobilenet_v3_small(weights=None)
            in_features = base_model.classifier[0].in_features
            # Extract features before the final classification head
            self.backbone = base_model.features
            self.avgpool = base_model.avgpool
            # Recreate classification head with num_classes
            self.head = nn.Sequential(
                nn.Linear(in_features, 1024),
                nn.Hardswish(),
                nn.Dropout(p=0.2),
                nn.Linear(1024, num_classes),
            )
            self.embedding_dim = in_features

        else:
            raise ValueError(
                f"Unsupported backbone: {backbone_name}. Choose from resnet50, resnet18, mobilenet_v3_small."
            )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract penultimate feature embedding vector (B, D)."""
        if self.backbone_name in ["resnet50", "resnet18"]:
            return self.backbone(x)
        else:
            features = self.backbone(x)
            pooled = self.avgpool(features)
            return torch.flatten(pooled, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning classification logits (B, num_classes)."""
        features = self.extract_features(x)
        return self.head(features)


def get_model(
    backbone_name: str = "resnet50",
    num_classes: int = DEFAULT_NUM_CLASSES,
    pretrained: bool = True,
) -> PetClassifierBackbone:
    """Build and return requested backbone model."""
    return PetClassifierBackbone(
        backbone_name=backbone_name,
        num_classes=num_classes,
        pretrained=pretrained,
    )
