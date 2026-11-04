"""
The nine ImageNet classifiers of the paper (Sec. 2.1), built exactly as the authors did:
the same architectures and the same pretrained weights (see paddle_models.py), and the same input
pipeline. InterpretDL's preprocess_image normalises every model's input with the ImageNet mean/std,
ViTs included.

GradCAM layers (benchmark-cv/run_expl.sh): layer4.2.relu for the ResNets (its last call is the block
output, which InterpretDL's overwriting hook keeps), lastconv.0 for MobileNet, features for VGG16.
GA/BT start layers (same script): ViT-S 2, ViT-B 4, ViT-L 8. MAE was only run in sweeps upstream
(2, 4, 6, 8 and the InterpretDL default 3), so it defaults to 4 like ViT-B; use --start_layers to sweep.
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .common import IMAGENET_MEAN, IMAGENET_STD
from .paddle_models import build, build_pretrained

CNN_METHODS = ["gradcam", "ig", "sg"]
VIT_METHODS = ["ig", "sg", "ga", "bt_head", "bt_token"]
BASELINE_METHODS = ["constant", "random_16", "random"]      # Table 1 only

VISION_MODEL_NAMES = ["mobilenet", "vgg16", "resnet50", "resnet101", "resnet152",
                      "vit_small", "vit_base", "vit_large", "mae_vit_base"]
START_LAYER = {"vit_small": 2, "vit_base": 4, "vit_large": 8, "mae_vit_base": 4}


@dataclass
class VisionModel:
    name: str
    family: str                                   # "cnn" | "vit"
    net: nn.Module
    gradcam_layer: nn.Module | None = None
    start_layer: int | None = None                # GA / BT
    methods: list = field(default_factory=list)
    mean: tuple = IMAGENET_MEAN
    std: tuple = IMAGENET_STD

    def normalize(self, x01: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.mean, device=x01.device).view(1, 3, 1, 1)
        std = torch.tensor(self.std, device=x01.device).view(1, 3, 1, 1)
        return (x01 - mean) / std

    def logits_from01(self, x01: torch.Tensor) -> torch.Tensor:
        return self.net(self.normalize(x01))


def load_vision_model(name: str, device, num_classes: int = 1000, pretrained: bool = True) -> VisionModel:
    """pretrained=False / num_classes=2 builds the from-scratch binary models used for SynScore."""
    if name not in VISION_MODEL_NAMES:
        raise ValueError(f"unknown vision model {name!r}; choose from {VISION_MODEL_NAMES}")
    net = build_pretrained(name) if pretrained else build(name, num_classes)
    if name.startswith("resnet"):
        m = VisionModel(name, "cnn", net, gradcam_layer=net.layer4[-1], methods=CNN_METHODS)
    elif name == "vgg16":
        m = VisionModel(name, "cnn", net, gradcam_layer=net.features, methods=CNN_METHODS)
    elif name == "mobilenet":
        m = VisionModel(name, "cnn", net, gradcam_layer=net.lastconv[0], methods=CNN_METHODS)
    else:
        m = VisionModel(name, "vit", net, start_layer=START_LAYER[name], methods=VIT_METHODS)
    m.net.eval().to(device)
    for p in m.net.parameters():
        p.requires_grad_(False)
    return m
