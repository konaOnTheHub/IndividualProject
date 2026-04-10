import os
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s

from api.labels import NUM_CLASSES

_DEFAULT_CKPT = Path(__file__).resolve().parent.parent / "best_efficientnetv2s_384.pth"


def resolve_checkpoint_path() -> Path:
    raw = os.environ.get("WASTE_MODEL_PATH")
    if raw:
        return Path(raw).expanduser().resolve()
    return _DEFAULT_CKPT.resolve()


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_efficientnet_v2_s(num_classes: int = NUM_CLASSES) -> nn.Module:
    weights = EfficientNet_V2_S_Weights.DEFAULT
    model = efficientnet_v2_s(weights=weights)
    if isinstance(model.classifier, nn.Sequential) and isinstance(
        model.classifier[-1], nn.Linear
    ):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    else:
        raise RuntimeError("Unexpected EfficientNetV2-S classifier structure")
    return model


def load_waste_classifier(
    ckpt_path: Path | None = None, device: torch.device | None = None
) -> tuple[nn.Module, torch.device]:
    path = ckpt_path or resolve_checkpoint_path()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    device = device or pick_device()
    model = build_efficientnet_v2_s()

    load_kw = {"map_location": device}
    try:
        ckpt = torch.load(path, **load_kw, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, **load_kw)

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    return model, device
