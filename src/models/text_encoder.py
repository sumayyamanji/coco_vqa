"""BERT-based text encoder — encodes the question into token embeddings."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from transformers import BertModel


class BERTTextEncoder(nn.Module):
    """Wraps ``bert-base-uncased`` and projects to model hidden_dim.

    Returns token_embeddings (all positions) first and cls_embedding ([CLS])
    second, mirroring the ordering of CLIPVisionEncoder for symmetry.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        model_cfg = config["model"]
        train_cfg = config.get("training", {})

        model_name: str = model_cfg.get("text_encoder", "bert-base-uncased")
        hidden_dim: int = model_cfg["hidden_dim"]
        freeze: bool = model_cfg.get("freeze_text", False)
        grad_ckpt: bool = train_cfg.get("gradient_checkpointing", False)

        self.bert = BertModel.from_pretrained(model_name)
        if freeze:
            for p in self.bert.parameters():
                p.requires_grad_(False)
        if grad_ckpt:
            self.bert.gradient_checkpointing_enable()

        bert_dim: int = self.bert.config.hidden_size
        self.proj = (
            nn.Linear(bert_dim, hidden_dim) if bert_dim != hidden_dim else nn.Identity()
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids:      (B, seq_len)
            attention_mask: (B, seq_len)
        Returns:
            token_embeddings: (B, seq_len, hidden_dim)
            cls_embedding:    (B, hidden_dim)
        """
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = self.norm(self.proj(out.last_hidden_state))  # (B, T, D)
        cls_embedding = token_embeddings[:, 0]                          # (B, D)
        return token_embeddings, cls_embedding


# Backward-compat alias
TextEncoder = BERTTextEncoder
