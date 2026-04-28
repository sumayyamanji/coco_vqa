"""BERT-based text encoder — encodes the question into token embeddings."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from transformers import BertModel


class TextEncoder(nn.Module):
    """Wraps a pretrained BERT model and projects its output to hidden_dim.

    Returns both the [CLS] token (sentence representation) and the full
    sequence of token embeddings for cross-attention with image patches.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        hidden_dim: int = 768,
        freeze_backbone: bool = False,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        if freeze_backbone:
            for p in self.bert.parameters():
                p.requires_grad_(False)
        if gradient_checkpointing:
            self.bert.gradient_checkpointing_enable()

        bert_dim = self.bert.config.hidden_size
        self.proj = nn.Linear(bert_dim, hidden_dim) if bert_dim != hidden_dim else nn.Identity()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids:      [B, seq_len]
            attention_mask: [B, seq_len]
        Returns:
            cls_token:     [B, hidden_dim]
            token_embeds:  [B, seq_len, hidden_dim]
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = self.proj(outputs.last_hidden_state[:, 0])
        token_embeds = self.proj(outputs.last_hidden_state)
        return cls_token, token_embeds
