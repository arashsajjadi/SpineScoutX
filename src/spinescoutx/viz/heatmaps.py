"""Grad-CAM and gradient-saliency evidence heatmaps for research models.

These utilities produce ``HxW`` heatmaps normalized to ``[0,1]`` highlighting
the image regions a classifier used. They support both the plain
``ImageClassifier`` (image + optional level/condition indices) and the
``AnatomyGuidedClassifier`` (image + anatomy prior + indices) via a uniform
``input_tensors`` payload.

Research-only — not diagnostic. Heatmaps are explanations of model behaviour,
not localizations of pathology.
"""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")

import cv2
import numpy as np
import torch
from torch import nn


def normalize_heatmap(arr: np.ndarray) -> np.ndarray:
    """Shift to ``>=0`` and scale so the max is 1; all-flat maps become zeros."""
    a = np.asarray(arr, dtype=np.float32)
    a = a - float(a.min())
    peak = float(a.max())
    if peak <= 1e-12:
        return np.zeros_like(a)
    return a / peak


def resize_heatmap(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a 2D heatmap to ``size`` = ``(height, width)`` with bilinear interp."""
    a = np.asarray(arr, dtype=np.float32)
    height, width = int(size[0]), int(size[1])
    return cv2.resize(a, (width, height), interpolation=cv2.INTER_LINEAR)


def _as_image_tensor(image: Any) -> torch.Tensor:
    """Return a ``(B, C, H, W)`` float tensor from a tensor/array image."""
    if not isinstance(image, torch.Tensor):
        image = torch.as_tensor(np.asarray(image, dtype=np.float32))
    image = image.float()
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if image.dim() != 4:
        raise ValueError(f"image must be (C,H,W) or (B,C,H,W), got {tuple(image.shape)}")
    return image


def _maybe_index(value: Any, batch: int, device: torch.device) -> torch.Tensor | None:
    """Coerce a level/condition index into a ``(B,)`` long tensor, or None."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        t = value.long().to(device)
        return t.view(-1)[:1].expand(batch) if t.numel() == 1 else t.view(-1)
    return torch.full((batch,), int(value), dtype=torch.long, device=device)


def _split_inputs(input_tensors: dict[str, Any] | tuple) -> dict[str, Any]:
    """Normalize a dict or positional tuple payload into a keyword dict.

    Tuple order: ``(image,)`` or ``(image, anatomy)`` or
    ``(image, level_idx, condition_idx)`` is not supported ambiguously, so a
    tuple is interpreted as ``(image, anatomy)`` with anatomy optional.
    """
    if isinstance(input_tensors, dict):
        return dict(input_tensors)
    if isinstance(input_tensors, tuple):
        if not input_tensors:
            raise ValueError("Empty input tuple for heatmap.")
        out: dict[str, Any] = {"image": input_tensors[0]}
        if len(input_tensors) > 1:
            out["anatomy"] = input_tensors[1]
        return out
    raise TypeError("input_tensors must be a dict or tuple.")


def _forward_logits(model: nn.Module, payload: dict[str, Any]) -> torch.Tensor:
    """Call ``model`` with whatever of image/anatomy/level/condition it accepts.

    Detects the anatomy-guided model by presence of an ``anatomy`` input and a
    forward signature accepting it; falls back to image-only otherwise.
    """
    image = _as_image_tensor(payload["image"])
    device = next(model.parameters()).device
    image = image.to(device)
    batch = image.shape[0]
    level_idx = _maybe_index(payload.get("level_idx"), batch, device)
    condition_idx = _maybe_index(payload.get("condition_idx"), batch, device)

    anatomy = payload.get("anatomy")
    use_anatomy = anatomy is not None and "anatomy" in model.forward.__code__.co_varnames
    if use_anatomy:
        anatomy_t = _as_image_tensor(anatomy).to(device)
        return model(image, anatomy_t, level_idx=level_idx, condition_idx=condition_idx)
    return model(image, level_idx=level_idx, condition_idx=condition_idx)


class GradCAM:
    """Grad-CAM heatmap generator using forward/backward hooks on a conv layer.

    Hooks are attached lazily on ``__call__`` and always removed afterwards so
    the model is left untouched. Works for both ``ImageClassifier`` and
    ``AnatomyGuidedClassifier`` (target the image-encoder conv layer returned by
    ``model.gradcam_target_layer()``).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

    def _save_activation(self, _module: nn.Module, _inp: Any, output: torch.Tensor) -> None:
        self._activations = output.detach()

    def _save_gradient(self, grad: torch.Tensor) -> None:
        self._gradients = grad.detach()

    def __call__(
        self,
        input_tensors: dict[str, Any] | tuple,
        class_idx: int | None = None,
    ) -> np.ndarray:
        payload = _split_inputs(input_tensors)
        image = _as_image_tensor(payload["image"])
        crop_hw = (int(image.shape[-2]), int(image.shape[-1]))

        was_training = self.model.training
        self.model.eval()
        forward_handle = self.target_layer.register_forward_hook(self._save_activation)
        backward_handle = self.target_layer.register_full_backward_hook(self._module_backward)
        try:
            self.model.zero_grad(set_to_none=True)
            logits = _forward_logits(self.model, payload)
            if self._activations is None:
                raise RuntimeError("Grad-CAM target layer produced no activation.")
            target_class = self._resolve_class(logits, class_idx)
            score = logits.gather(1, target_class.view(-1, 1)).sum()
            score.backward()
            cam = self._build_cam()
        finally:
            forward_handle.remove()
            backward_handle.remove()
            self._activations = None
            self._gradients = None
            if was_training:
                self.model.train()
        return resize_heatmap(normalize_heatmap(cam), crop_hw)

    def _module_backward(self, _m: nn.Module, _gi: Any, grad_output: tuple) -> None:
        self._save_gradient(grad_output[0])

    def _resolve_class(self, logits: torch.Tensor, class_idx: int | None) -> torch.Tensor:
        if class_idx is None:
            return logits.argmax(dim=1)
        return torch.full(
            (logits.shape[0],), int(class_idx), dtype=torch.long, device=logits.device
        )

    def _build_cam(self) -> np.ndarray:
        if self._activations is None or self._gradients is None:
            raise RuntimeError("Grad-CAM did not capture activations and gradients.")
        activations = self._activations
        gradients = self._gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1)
        cam = torch.relu(cam)
        return cam[0].cpu().numpy().astype(np.float32)


def gradient_saliency(
    model: nn.Module,
    input_tensors: dict[str, Any] | tuple,
    class_idx: int | None = None,
) -> np.ndarray:
    """Vanilla input-gradient saliency (fallback when Grad-CAM is unavailable).

    Returns an ``HxW`` heatmap normalized to ``[0,1]`` (max over channels of the
    absolute input gradient).
    """
    payload = _split_inputs(input_tensors)
    image = _as_image_tensor(payload["image"])
    device = next(model.parameters()).device
    image = image.to(device).clone().requires_grad_(True)
    payload = dict(payload)
    payload["image"] = image

    was_training = model.training
    model.eval()
    try:
        model.zero_grad(set_to_none=True)
        logits = _forward_logits(model, payload)
        if class_idx is None:
            target = logits.argmax(dim=1)
        else:
            target = torch.full(
                (logits.shape[0],), int(class_idx), dtype=torch.long, device=logits.device
            )
        score = logits.gather(1, target.view(-1, 1)).sum()
        score.backward()
        grad = image.grad
        if grad is None:
            raise RuntimeError("Saliency could not compute an input gradient.")
        sal = grad.detach().abs().amax(dim=1)[0].cpu().numpy().astype(np.float32)
    finally:
        if was_training:
            model.train()
    return normalize_heatmap(sal)


def make_heatmap(
    model: nn.Module,
    sample: dict[str, Any],
    class_idx: int | None = None,
    method: str = "gradcam",
) -> np.ndarray:
    """Convenience: produce a heatmap for one ``sample`` dict from a dataset.

    ``sample`` may contain ``image``, ``anatomy``, ``level_idx``,
    ``condition_idx``. ``method`` is "gradcam" (default) or "saliency".
    """
    payload: dict[str, Any] = {"image": sample["image"]}
    for key in ("anatomy", "level_idx", "condition_idx"):
        if key in sample and sample[key] is not None:
            payload[key] = sample[key]

    if method == "gradcam":
        target_layer = model.gradcam_target_layer()
        cam = GradCAM(model, target_layer)
        return cam(payload, class_idx=class_idx)
    if method == "saliency":
        return gradient_saliency(model, payload, class_idx=class_idx)
    raise ValueError(f"Unknown heatmap method: {method!r} (use 'gradcam' or 'saliency').")


__all__ = [
    "GradCAM",
    "gradient_saliency",
    "make_heatmap",
    "normalize_heatmap",
    "resize_heatmap",
]
