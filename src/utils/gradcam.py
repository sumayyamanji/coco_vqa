"""GradCAM — gradient-weighted class activation maps for VQA vision encoder layers."""
from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """Registers forward/backward hooks on a target layer and computes spatial
    saliency maps for a given (or predicted) answer class.

    Supports both:
    - **String lookup**: ``GradCAM(model, "vision_encoder.backbone.vision_model.encoder.layers.23")``
    - **Direct module**: ``GradCAM(model, model.vision_encoder.backbone.vision_model.encoder.layers[-1])``

    Both transformer patch tokens (B, N+1, D) and CNN feature maps (B, C, H, W)
    are handled automatically.

    Usage
    -----
    ::
        gcam = GradCAM(model, "vision_encoder.backbone.vision_model.encoder.layers.23")
        heatmap = gcam.compute(image_tensor, question_ids, attention_mask, target_class=42)
        overlay = gcam.overlay(pil_image, heatmap)

    Legacy call style (backward compat)::
        gcam = GradCAM(model, target_layer=some_module)
        heatmap = gcam(pixel_values, input_ids, attention_mask, target_class=42)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: Union[str, torch.nn.Module],
    ) -> None:
        self.model = model
        self._activations: Optional[torch.Tensor] = None
        self._gradients:   Optional[torch.Tensor] = None
        self._hooks: list = []

        # Resolve string → module
        if isinstance(target_layer, str):
            named = dict(model.named_modules())
            if target_layer not in named:
                raise ValueError(
                    f"Layer '{target_layer}' not found in model. "
                    f"Available names (first 20): {list(named)[:20]}"
                )
            target_module = named[target_layer]
        else:
            target_module = target_layer

        self._hooks.append(
            target_module.register_forward_hook(self._save_activation)
        )
        self._hooks.append(
            target_module.register_full_backward_hook(self._save_gradient)
        )

    # ------------------------------------------------------------------
    # Hook callbacks
    # ------------------------------------------------------------------

    def _save_activation(self, _module: Any, _inp: Any, output: Any) -> None:
        a = output[0] if isinstance(output, tuple) else output
        self._activations = a.detach()

    def _save_gradient(self, _module: Any, _gin: Any, gout: Any) -> None:
        g = gout[0] if isinstance(gout, tuple) else gout
        if g is not None:
            self._gradients = g.detach()

    # ------------------------------------------------------------------
    # compute — new dict-batch API
    # ------------------------------------------------------------------

    def compute(
        self,
        image_tensor: torch.Tensor,
        question_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """Compute a Grad-CAM heatmap resized to the input image resolution.

        Args:
            image_tensor:    ``(1, C, H, W)`` normalised float tensor
            question_ids:    ``(1, seq_len)`` token ids; may be None
            attention_mask:  ``(1, seq_len)``; may be None
            target_class:    answer vocab index to explain;
                             uses argmax of ``answer_logits`` when None

        Returns:
            Heatmap as a ``(H, W)`` float32 numpy array in ``[0, 1]``.
        """
        batch: Dict[str, Any] = {
            "image_tensor":   image_tensor,
            "question_ids":   question_ids,
            "attention_mask": attention_mask,
        }

        was_training = self.model.training
        self.model.eval()

        try:
            out = self.model(batch)
            logits = out["answer_logits"]  # (1, vocab_size)

            if target_class is None:
                target_class = int(logits.argmax(dim=-1).item())

            self.model.zero_grad()
            logits[0, target_class].backward()
        finally:
            if was_training:
                self.model.train()

        return self._build_heatmap(image_tensor)

    # ------------------------------------------------------------------
    # overlay — jet colourmap blend
    # ------------------------------------------------------------------

    def overlay(
        self,
        image_pil: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.5,
    ) -> Image.Image:
        """Blend a Grad-CAM heatmap over a PIL image using the jet colourmap.

        Args:
            image_pil: original image (any size)
            heatmap:   ``(H, W)`` float32 array in ``[0, 1]`` — typically the
                       output of :meth:`compute` (already at image resolution)
            alpha:     heatmap blend weight (0 = only image, 1 = only heatmap)

        Returns:
            Blended PIL Image (RGB).
        """
        import matplotlib.cm as _cm

        img_np = np.asarray(image_pil.convert("RGB")).astype(np.float32) / 255.0
        h, w = img_np.shape[:2]

        # Resize heatmap to image dimensions (bicubic)
        hm_pil = Image.fromarray(
            (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
        ).resize((w, h), Image.BICUBIC)
        hm_np = np.asarray(hm_pil).astype(np.float32) / 255.0

        # Apply jet colourmap (drop alpha channel → RGB float)
        jet = _cm.get_cmap("jet")
        hm_colored = jet(hm_np)[:, :, :3].astype(np.float32)

        # Alpha blend and convert to uint8
        blended = np.clip(
            (1.0 - alpha) * img_np + alpha * hm_colored, 0.0, 1.0
        )
        return Image.fromarray((blended * 255).astype(np.uint8))

    # ------------------------------------------------------------------
    # remove hooks
    # ------------------------------------------------------------------

    def remove_hooks(self) -> None:
        """Deregister all forward/backward hooks (call when done)."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    # Internal: build heatmap from captured activations / gradients
    # ------------------------------------------------------------------

    def _build_heatmap(self, image_tensor: torch.Tensor) -> np.ndarray:
        acts  = self._activations
        grads = self._gradients

        if acts is None or grads is None:
            raise RuntimeError(
                "Hooks did not capture activations/gradients. "
                "Ensure a forward+backward pass was run after hook registration."
            )

        H = image_tensor.shape[-2]
        W = image_tensor.shape[-1]

        if acts.dim() == 3:
            # Transformer: (B, seq, D) — skip CLS token at position 0
            a = acts[0, 1:]   # (N, D)
            g = grads[0, 1:]  # (N, D)
            weights = g.mean(dim=-1)                            # (N,)
            cam = F.relu((weights.unsqueeze(-1) * a).sum(dim=-1))  # (N,)
        elif acts.dim() == 4:
            # CNN: (B, C, h, w)
            weights = grads.mean(dim=(2, 3), keepdim=True)     # (B, C, 1, 1)
            cam = F.relu((weights * acts).sum(dim=1))           # (B, h, w)
            cam = cam.squeeze(0)                                 # (h, w)
        else:
            raise RuntimeError(f"Unexpected activation shape: {acts.shape}")

        cam_np = cam.detach().cpu().float().numpy()

        # Normalise to [0, 1]
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)

        # Reshape 1-D transformer output to 2-D patch grid
        if cam_np.ndim == 1:
            side = int(np.sqrt(cam_np.shape[0]))
            cam_np = cam_np[: side * side].reshape(side, side)

        # Resize to original image dimensions
        hm_pil = Image.fromarray((cam_np * 255).astype(np.uint8)).resize(
            (W, H), Image.BICUBIC
        )
        return np.asarray(hm_pil).astype(np.float32) / 255.0

    # ------------------------------------------------------------------
    # Legacy __call__ (backward compat)
    # ------------------------------------------------------------------

    def __call__(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """Legacy call signature — wraps :meth:`compute`.

        Note: this builds a batch dict and calls ``model(batch)`` internally,
        so the model must accept the dict-based forward signature.
        """
        return self.compute(pixel_values, input_ids, attention_mask, target_class)
