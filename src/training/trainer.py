"""Trainer — orchestrates the full training and validation loop."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..evaluation.metrics import soft_accuracy, per_type_accuracy
from ..utils.checkpoint import save_checkpoint
from .losses import TotalLoss
from .scheduler import get_cosine_schedule_with_warmup


# answer-type string → integer class index
_TYPE_MAP: Dict[str, int] = {"yes/no": 0, "number": 1, "other": 2}
_TYPE_NAMES: List[str] = ["yes/no", "number", "other"]


class Trainer:
    """Encapsulates one full training run.

    Supports:
      - Mixed-precision training (fp16) via torch.cuda.amp, guarded to CUDA
      - Cosine LR schedule with linear warmup
      - Gradient clipping
      - Batch-level W&B logging every 50 steps (loss, lr, grad_norm)
      - Epoch-level VQA soft accuracy (overall + per answer type)
      - Confusion-matrix logging for the type classifier
      - Checkpoint rotation with keep-last-N; separate best-model save
      - Clean progress table printed to stdout each epoch

    Args:
        model:        nn.Module — the VQAModel
        config:       parsed config.yaml dict
        train_loader: DataLoader for the training split
        val_loader:   DataLoader for the validation split
        device:       torch.device
        wandb_run:    optional wandb Run (or WandbLogger) — duck-typed,
                      must expose a `.log(metrics_dict)` method
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        wandb_run: Optional[Any] = None,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.wandb_run = wandb_run

        t_cfg = config["training"]
        self.use_fp16: bool = t_cfg["fp16"] and device.type == "cuda"
        self.grad_clip: float = float(t_cfg["grad_clip"])

        self.criterion = TotalLoss(config)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(t_cfg["lr"]),
            weight_decay=float(t_cfg["weight_decay"]),
        )

        total_steps = t_cfg["epochs"] * len(train_loader)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            warmup_steps=t_cfg["warmup_steps"],
            total_steps=total_steps,
        )
        self.scaler = GradScaler("cuda", enabled=self.use_fp16)

        self.ckpt_dir = Path(config["paths"]["checkpoint_dir"])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self._global_step: int = 0
        self.best_ckpt_path: Optional[Path] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_epoch(self) -> Dict[str, float]:
        """Run one full pass over the training set.

        Returns:
            dict with keys: loss, vqa_loss, type_loss (epoch averages)
        """
        self.model.train()
        total_loss = total_vqa = total_type = 0.0

        pbar = tqdm(self.train_loader, desc="  train", leave=False, dynamic_ncols=True)
        for batch in pbar:
            batch_dev = self._to_device(batch)
            self.optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=self.use_fp16):
                out = self.model(batch_dev)
                type_labels = self._type_labels(batch["answer_type"], self.device)
                loss, loss_parts = self.criterion(
                    out["answer_logits"],
                    out["answer_type_logits"],
                    batch_dev["answer_scores"],
                    type_labels,
                )

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            loss_val = float(loss)
            total_loss += loss_val
            total_vqa += loss_parts["vqa_loss"]
            total_type += loss_parts["type_loss"]
            self._global_step += 1

            pbar.set_postfix(loss=f"{loss_val:.4f}")

            if self._global_step % 50 == 0:
                self._log({
                    "train/batch_loss": loss_val,
                    "train/vqa_loss": loss_parts["vqa_loss"],
                    "train/type_loss": loss_parts["type_loss"],
                    "train/lr": self.scheduler.get_last_lr()[0],
                    "train/grad_norm": grad_norm,
                })

        n = max(len(self.train_loader), 1)
        return {"loss": total_loss / n, "vqa_loss": total_vqa / n, "type_loss": total_type / n}

    @torch.no_grad()
    def validate(self) -> Dict[str, Any]:
        """Run full validation set.

        Returns:
            dict with keys:
              vqa_accuracy        — overall soft accuracy
              per_type_accuracy   — {yes/no: float, number: float, other: float}
              type_accuracy       — accuracy of the type classifier itself
        """
        self.model.eval()

        all_ans_logits: List[torch.Tensor] = []
        all_ans_scores: List[torch.Tensor] = []
        all_type_strs: List[str] = []
        all_type_preds: List[int] = []

        for batch in tqdm(self.val_loader, desc="  val  ", leave=False, dynamic_ncols=True):
            batch_dev = self._to_device(batch)
            out = self.model(batch_dev)

            all_ans_logits.append(out["answer_logits"].cpu())
            all_ans_scores.append(batch_dev["answer_scores"].cpu())
            all_type_strs.extend(batch["answer_type"])
            all_type_preds.extend(
                out["answer_type_logits"].argmax(dim=-1).cpu().tolist()
            )

        logits = torch.cat(all_ans_logits, dim=0)
        scores = torch.cat(all_ans_scores, dim=0)

        vqa_acc = soft_accuracy(logits, scores)
        type_acc_map = per_type_accuracy(logits, scores, all_type_strs)

        # Type-classifier accuracy
        y_true = [_TYPE_MAP.get(t, 2) for t in all_type_strs]
        type_clf_acc = (
            sum(p == t for p, t in zip(all_type_preds, y_true)) / max(len(y_true), 1)
        )

        # Confusion matrix → wandb
        self._log_confusion_matrix(y_true, all_type_preds)

        return {
            "vqa_accuracy": vqa_acc,
            "per_type_accuracy": type_acc_map,
            "type_classifier_accuracy": type_clf_acc,
        }

    def train(self, num_epochs: int, start_epoch: int = 0) -> None:
        """Full training loop.

        Args:
            num_epochs:  number of epochs to run
            start_epoch: epoch offset (set > 0 when resuming a checkpoint)
        """
        best_acc = 0.0
        t_cfg = self.config["training"]
        log_cfg = self.config["logging"]

        for epoch in range(start_epoch + 1, start_epoch + num_epochs + 1):
            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            acc = val_metrics["vqa_accuracy"]
            is_best = acc > best_acc

            # Save periodic checkpoint
            save_checkpoint(
                self.model, self.optimizer, epoch, self.ckpt_dir,
                keep_last_n=log_cfg["keep_last_n"],
                extra={"val_vqa_accuracy": acc},
            )

            # Save best model separately (always overwrite)
            if is_best:
                best_acc = acc
                best_path = self.ckpt_dir / "best_model.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_vqa_accuracy": acc,
                }, best_path)
                self.best_ckpt_path = best_path

            # W&B epoch-level logging
            pt = val_metrics.get("per_type_accuracy", {})
            self._log({
                "epoch": epoch,
                "train/loss": train_metrics["loss"],
                "train/vqa_loss": train_metrics["vqa_loss"],
                "train/type_loss": train_metrics["type_loss"],
                "val/vqa_accuracy": acc,
                "val/type_clf_accuracy": val_metrics["type_classifier_accuracy"],
                **{f"val/{k}": v for k, v in pt.items()},
            })

            self._print_epoch(epoch, start_epoch + num_epochs, train_metrics, val_metrics, is_best)

        print(f"\nTraining complete — best val VQA accuracy: {best_acc:.4f}")
        if self.best_ckpt_path:
            print(f"Best checkpoint: {self.best_ckpt_path}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_device(self, batch: dict) -> dict:
        """Move all tensor values in *batch* to self.device, leave others as-is."""
        return {
            k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    @staticmethod
    def _type_labels(type_strs: List[str], device: torch.device) -> torch.Tensor:
        return torch.tensor(
            [_TYPE_MAP.get(t, 2) for t in type_strs],
            dtype=torch.long, device=device,
        )

    def _log(self, metrics: dict, step: Optional[int] = None) -> None:
        if self.wandb_run is None:
            return
        try:
            self.wandb_run.log(metrics, step=step)
        except Exception:
            pass

    def _log_confusion_matrix(self, y_true: List[int], y_pred: List[int]) -> None:
        if self.wandb_run is None:
            return
        try:
            import wandb
            cm = wandb.plot.confusion_matrix(
                y_true=y_true,
                preds=y_pred,
                class_names=_TYPE_NAMES,
            )
            self.wandb_run.log({"val/type_confusion_matrix": cm})
        except Exception:
            pass

    @staticmethod
    def _print_epoch(
        epoch: int,
        total_epochs: int,
        train_m: dict,
        val_m: dict,
        is_best: bool,
    ) -> None:
        bar = "=" * 72
        pt = val_m.get("per_type_accuracy", {})
        yn = pt.get("yes/no", float("nan"))
        nu = pt.get("number", float("nan"))
        ot = pt.get("other", float("nan"))
        best_tag = " [BEST]" if is_best else ""
        print(bar)
        print(f"  Epoch {epoch:03d}/{total_epochs:03d}")
        print(
            f"  Train  loss={train_m['loss']:.4f}"
            f"  vqa={train_m['vqa_loss']:.4f}"
            f"  type={train_m['type_loss']:.4f}"
        )
        print(
            f"  Val    vqa_acc={val_m['vqa_accuracy']:.4f}"
            f"  yes/no={yn:.4f}  number={nu:.4f}  other={ot:.4f}"
            + best_tag
        )
        print(bar)
