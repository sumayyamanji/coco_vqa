"""W&B wrapper with graceful no-op fallback and rich logging helpers."""
from __future__ import annotations

import io
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


class WandbLogger:
    """Wraps wandb with a no-op fallback when wandb is unavailable or disabled.

    New API (config-first)
    ----------------------
    ::
        logger = WandbLogger(config, project="coco-vqa", run_name="run-001")

    Legacy API (project-first) still works via keyword arguments::
        logger = WandbLogger(project="coco-vqa", config=cfg)

    All ``log_*`` methods silently do nothing when wandb is unavailable.
    """

    def __init__(
        self,
        config: dict | None = None,
        project: str | None = None,
        run_name: str | None = None,
        enabled: bool = True,
    ) -> None:
        self._run = None

        if not enabled:
            return

        if run_name is None:
            ts = datetime.now().strftime("%m%d-%H%M")
            mode = (config or {}).get("mode", "multimodal") if isinstance(config, dict) else "run"
            run_name = f"{mode}-{ts}"

        try:
            import wandb
            self._run = wandb.init(
                project=project or "coco-vqa",
                name=run_name,
                config=config or {},
            )
        except Exception as exc:
            print(f"[WandbLogger] W&B unavailable ({exc}). Logging disabled.")

    # ------------------------------------------------------------------
    # Core scalar logging
    # ------------------------------------------------------------------

    def log(self, metrics: Dict[str, Any], step: int | None = None) -> None:
        """Log a dict of scalars (legacy method, kept for backward compat)."""
        if self._run is None:
            return
        try:
            self._run.log(metrics, step=step)
        except Exception:
            pass

    def log_metrics(self, metrics_dict: Dict[str, Any], step: int) -> None:
        """Log a dict of scalar metrics at the given global step."""
        self.log(metrics_dict, step=step)

    # ------------------------------------------------------------------
    # Image logging
    # ------------------------------------------------------------------

    def log_image(
        self,
        name: str,
        pil_image: Any,
        step: int,
        caption: str = "",
    ) -> None:
        """Log a single PIL Image (or numpy array) to W&B.

        Args:
            name:      key shown in the W&B media panel
            pil_image: PIL.Image or H×W×C uint8 numpy array
            step:      global training step
            caption:   optional image caption
        """
        if self._run is None:
            return
        try:
            import wandb
            self._run.log(
                {name: wandb.Image(pil_image, caption=caption)},
                step=step,
            )
        except Exception:
            pass

    def log_attention_heatmap(
        self,
        image: Any,
        heatmap: np.ndarray,
        question: str,
        answer: str,
        step: int,
        alpha: float = 0.5,
    ) -> None:
        """Overlay a patch-attention heatmap on the original image and log to W&B.

        Args:
            image:    PIL Image or H×W×C uint8 numpy array
            heatmap:  H×W float32 array in [0, 1]; will be resized to match image
            question: question string (used in caption)
            answer:   predicted answer (used in caption)
            step:     global step
            alpha:    heatmap blend weight
        """
        if self._run is None:
            return
        try:
            import wandb
            from PIL import Image as _PILImage
            import matplotlib.cm as _cm

            # Normalise image to float [0,1]
            if isinstance(image, _PILImage.Image):
                img_np = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
            else:
                img_np = np.asarray(image).astype(np.float32) / 255.0
                if img_np.max() > 1.0:
                    img_np /= 255.0

            h, w = img_np.shape[:2]

            # Resize heatmap to image size (bicubic via PIL)
            hm_pil = _PILImage.fromarray(
                (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
            ).resize((w, h), _PILImage.BICUBIC)
            hm_np = np.asarray(hm_pil).astype(np.float32) / 255.0

            # Apply jet colourmap (drop alpha channel)
            jet = _cm.get_cmap("jet")
            hm_colored = jet(hm_np)[:, :, :3].astype(np.float32)

            # Alpha blend
            blended = np.clip(
                (1.0 - alpha) * img_np + alpha * hm_colored, 0.0, 1.0
            )
            blended_u8 = (blended * 255).astype(np.uint8)

            caption = f"Q: {question[:120]}\nA: {answer}"
            self._run.log(
                {"attention_heatmap": wandb.Image(blended_u8, caption=caption)},
                step=step,
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------------

    def log_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: Sequence[str],
        step: int,
    ) -> None:
        """Log a confusion matrix using wandb.plot.confusion_matrix.

        Args:
            cm:           (N, N) int numpy array; rows = true, cols = predicted
            class_names:  list of N class-name strings
            step:         global step
        """
        if self._run is None:
            return
        try:
            import wandb

            # Expand the count matrix into flat (y_true, y_pred) lists
            y_true: List[str] = []
            y_pred: List[str] = []
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    count = int(cm[i, j])
                    y_true.extend([class_names[i]] * count)
                    y_pred.extend([class_names[j]] * count)

            self._run.log(
                {
                    "confusion_matrix": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=y_true,
                        preds=y_pred,
                        class_names=list(class_names),
                    )
                },
                step=step,
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------

    def log_comparison_table(
        self,
        text_only_metrics: Dict[str, Any],
        multimodal_metrics: Dict[str, Any],
        step: int,
        image_only_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a W&B Table comparing per-type accuracy across model variants.

        Args:
            text_only_metrics:   {"accuracy": float, "per_type": {type: float}}
            multimodal_metrics:  same structure
            step:                global step
            image_only_metrics:  optional; same structure
        """
        if self._run is None:
            return
        try:
            import wandb

            columns = ["Mode", "Overall", "Yes/No", "Number", "Other"]

            def _row(label: str, res: Dict[str, Any]) -> List:
                pt = res.get("per_type", {})
                return [
                    label,
                    round(res.get("accuracy", 0.0), 4),
                    round(pt.get("yes/no", 0.0), 4),
                    round(pt.get("number", 0.0), 4),
                    round(pt.get("other", 0.0), 4),
                ]

            rows = [_row("Text Only", text_only_metrics)]
            if image_only_metrics is not None:
                rows.append(_row("Image Only", image_only_metrics))
            rows.append(_row("Multimodal", multimodal_metrics))

            table = wandb.Table(columns=columns, data=rows)
            self._run.log({"model_comparison": table}, step=step)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Model watching
    # ------------------------------------------------------------------

    def watch(self, model: Any, log_freq: int = 100) -> None:
        """Call wandb.watch on the model to log gradient histograms."""
        if self._run is None:
            return
        try:
            import wandb
            wandb.watch(model, log_freq=log_freq)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def finish(self) -> None:
        """Close the W&B run."""
        if self._run is not None:
            try:
                self._run.finish()
            except Exception:
                pass
