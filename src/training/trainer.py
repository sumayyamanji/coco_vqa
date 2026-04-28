"""Trainer — orchestrates the full training and validation loop."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.checkpoint import save_checkpoint
from ..utils.wandb_logger import WandbLogger
from .losses import VQALoss
from .scheduler import build_scheduler


class Trainer:
    """Encapsulates one full training run.

    Supports:
      - Mixed-precision training (fp16) via torch.cuda.amp
      - Gradient clipping
      - Linear warm-up + cosine LR decay
      - Periodic checkpointing with keep-last-N rotation
      - Optional W&B logging
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: dict,
        device: torch.device,
        logger: Optional[WandbLogger] = None,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.logger = logger

        self.criterion = VQALoss(label_smoothing=cfg["training"]["label_smoothing"])
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["training"]["lr"],
            weight_decay=cfg["training"]["weight_decay"],
        )
        total_steps = cfg["training"]["epochs"] * len(train_loader)
        self.scheduler = build_scheduler(
            self.optimizer,
            warmup_steps=cfg["training"]["warmup_steps"],
            total_steps=total_steps,
        )
        self.scaler = GradScaler(enabled=cfg["training"]["fp16"])
        self.ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self) -> None:
        for epoch in range(1, self.cfg["training"]["epochs"] + 1):
            train_loss = self._train_epoch(epoch)
            val_loss, val_acc = self._val_epoch(epoch)
            print(
                f"Epoch {epoch:03d} | train_loss={train_loss:.4f}"
                f" | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
            )
            if self.logger:
                self.logger.log(
                    {"train/loss": train_loss, "val/loss": val_loss, "val/acc": val_acc},
                    step=epoch,
                )
            if epoch % self.cfg["logging"]["save_every"] == 0:
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    self.ckpt_dir,
                    keep_last_n=self.cfg["logging"]["keep_last_n"],
                )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Train {epoch}", leave=False)
        for batch in pbar:
            self.optimizer.zero_grad()
            with autocast(enabled=self.cfg["training"]["fp16"]):
                logits = self.model(
                    pixel_values=batch["image"].to(self.device),
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                )
                loss = self.criterion(logits, batch["soft_scores"].to(self.device))
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg["training"]["grad_clip"]
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def _val_epoch(self, epoch: int):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        for batch in tqdm(self.val_loader, desc=f"Val   {epoch}", leave=False):
            logits = self.model(
                pixel_values=batch["image"].to(self.device),
                input_ids=batch["input_ids"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
            )
            loss = self.criterion(logits, batch["soft_scores"].to(self.device))
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds.cpu() == batch["label"]).sum().item()
            total += len(batch["label"])
        return total_loss / len(self.val_loader), correct / max(total, 1)
