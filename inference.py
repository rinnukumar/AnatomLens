"""CPU inference and spatial explanations for the original checkpoints."""

import hashlib
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps

from models import BaselineCNN, SmallViT

ROOT = Path(__file__).resolve().parent
NAMES = (
    "Bladder", "Femur-left", "Femur-right", "Heart", "Kidney-left",
    "Kidney-right", "Liver", "Lung-left", "Lung-right", "Pancreas", "Spleen",
)
MEAN = 0.4680533955970271
STD = 0.28575827805106974
DATA_SHA256 = "7c39150fd57085477a9f0b7f1dcec76eae366f578f4e600653ddc15d46aab6e0"


def load_models():
    torch.set_num_threads(2)
    models = {"CNN": BaselineCNN(), "ViT": SmallViT()}
    for name, model in models.items():
        path = ROOT / "checkpoints" / f"{name.lower()}.pth"
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.eval().requires_grad_(False)
    return models


def load_cases(path=None):
    full = path or os.getenv("ANATOMLENS_DATA")
    if full:
        with open(full, "rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != DATA_SHA256:
            raise ValueError("Dataset checksum differs from the research OrganAMNIST-64 archive.")
        with np.load(full, allow_pickle=False) as data:
            images = data["test_images"]
            return images, data["test_labels"].ravel(), np.arange(len(images))

    with np.load(ROOT / "data" / "explorer_cases.npz", allow_pickle=False) as data:
        return data["images"], data["labels"], data["indices"]


def preprocess(images):
    return (torch.from_numpy(images.astype(np.float32)).unsqueeze(1) / 255 - MEAN) / STD


def probabilities(model, inputs):
    with torch.inference_mode():
        return model(inputs).softmax(dim=1).numpy()


def gradcam(model, inputs, target):
    with torch.enable_grad():
        image = inputs.detach().requires_grad_(True)
        # Third convolution, as in the original evaluation notebook.
        activation = model.features[:9](image)
        logits = model.classifier(model.pool(model.features[9:](activation)).flatten(1))
        gradient, = torch.autograd.grad(logits[0, target], activation)
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        return resize_map((activation * weights).sum(dim=1, keepdim=True).relu())


@torch.inference_mode()
def attention_rollout(model, inputs):
    tokens = model.patch_embedding(inputs).flatten(2).transpose(1, 2)
    tokens = torch.cat((model.cls_token.expand(tokens.size(0), -1, -1), tokens), dim=1)
    tokens = model.dropout(tokens + model.pos_embedding)
    identity = torch.eye(tokens.size(1), dtype=tokens.dtype)
    rollout = identity
    for layer in model.transformer.layers:
        normalized = layer.norm1(tokens)
        _, attention = layer.self_attn(
            normalized, normalized, normalized, need_weights=True, average_attn_weights=False
        )
        attention = attention.mean(dim=1)[0] + identity
        rollout = (attention / attention.sum(dim=-1, keepdim=True)) @ rollout
        tokens = layer(tokens)
    torch.testing.assert_close(
        model.classifier(model.norm(tokens[:, 0])), model(inputs), atol=1e-4, rtol=1e-4
    )
    return resize_map(rollout[0, 1:].reshape(1, 1, 8, 8))


def resize_map(mask):
    mask = F.interpolate(mask, size=(64, 64), mode="bilinear", align_corners=False)
    mask = mask.detach().squeeze().numpy()
    return (mask - mask.min()) / max(float(np.ptp(mask)), 1e-8)


def overlay(image, mask):
    gray = np.repeat(image[..., None] / 255, 3, axis=-1)
    colors = colormaps["inferno"](mask)[..., :3]
    opacity = 0.65 * mask[..., None]
    return ((gray * (1 - opacity) + colors * opacity) * 255).astype(np.uint8)
