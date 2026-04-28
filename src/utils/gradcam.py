"""GradCAM — gradient-weighted class activation maps for vision encoder layers."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """Registers forward/backward hooks on a target conv/attention layer and
    computes a spatial saliency map for the predicted (or a given) class.

    Usage::
        gcam = GradCAM(model, target_layer=model.vision_encoder.backbone.vision_model.encoder.layers[-1])
        heatmap = gcam(pixel_values, input_ids, attention_mask, target_class=42)
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _input, output) -> None:
        self._activations = output.detach()

    def _save_gradient(self, _module, _grad_input, grad_output) -> None:
        self._gradients = grad_output[0].detach()

    def __call__(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Returns:
            heatmap: [H, W] float32 numpy array normalised to [0, 1]
        """
        self.model.eval()
        logits = self.model(pixel_values, input_ids, attention_mask)
        if target_class is None:
            target_class = int(logits.argmax(dim=-1).item())

        self.model.zero_grad()
        logits[0, target_class].backward()

        # Pool gradients across the sequence/spatial dimension
        weights = self._gradients.mean(dim=list(range(1, self._gradients.ndim)), keepdim=True)
        cam = (weights * self._activations).sum(dim=-1)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalise
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        return cam.astype(np.float32)
