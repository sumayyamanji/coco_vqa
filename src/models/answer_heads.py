"""Answer prediction heads — specialized classifiers and a generative decoder."""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Kept from original: general gated MLP classifier
# ---------------------------------------------------------------------------

class AnswerClassifier(nn.Module):
    """Gated two-layer MLP from fused embedding → answer logits (original head)."""

    def __init__(
        self,
        hidden_dim: int = 768,
        num_answer_classes: int = 3129,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.proj = nn.Linear(hidden_dim * 2, num_answer_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.gate(x))


# ---------------------------------------------------------------------------
# Answer-type routing heads
# ---------------------------------------------------------------------------

class YesNoHead(nn.Module):
    """Binary classifier for yes/no questions."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, 2) logits — index 0 = no, 1 = yes."""
        return self.linear(x)


class NumberHead(nn.Module):
    """Classifier over integer answers 0-49 for numeric questions."""

    NUM_CLASSES: int = 50

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, self.NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, 50) logits."""
        return self.linear(x)


class OpenEndedHead(nn.Module):
    """Two-layer MLP for open-ended / other-type answers over the full vocab."""

    def __init__(self, hidden_dim: int, num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, num_classes) logits."""
        return self.mlp(x)


class AnswerTypeClassifier(nn.Module):
    """3-way classifier that predicts the answer type for routing.

    Class mapping:
        0 → yes/no
        1 → number
        2 → other (open-ended)
    """

    TYPE_YESNO = 0
    TYPE_NUMBER = 1
    TYPE_OTHER = 2

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.clf = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, 3) logits."""
        return self.clf(x)


# ---------------------------------------------------------------------------
# Generative head — 4-layer autoregressive transformer decoder
# ---------------------------------------------------------------------------

class GenerativeHead(nn.Module):
    """Small autoregressive decoder that generates answer tokens from fused features.

    The fused context vector acts as the single-token memory for cross-attention.
    Vocabulary uses the VQA answer vocab (``vocab_size`` tokens) plus two
    special tokens: BOS = vocab_size, EOS = vocab_size + 1.

    During training pass ``target_ids`` for teacher-forced logit computation.
    During inference call ``generate()`` for greedy or beam-search decoding.
    """

    BOS: int = -2  # set to vocab_size in __init__
    EOS: int = -1  # set to vocab_size + 1 in __init__

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        max_len: int = 10,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.BOS = vocab_size
        self.EOS = vocab_size + 1
        full_vocab = vocab_size + 2  # include BOS + EOS

        self.token_emb = nn.Embedding(full_vocab, hidden_dim)
        self.pos_emb = nn.Embedding(max_len + 2, hidden_dim)

        # Project fused vector to a 1-token memory sequence for cross-attention
        self.context_proj = nn.Linear(hidden_dim, hidden_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # pre-norm (more stable)
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(hidden_dim, full_vocab)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, T) → (B, T, D)"""
        T = tokens.size(1)
        pos = torch.arange(T, device=tokens.device).unsqueeze(0)
        return self.token_emb(tokens) + self.pos_emb(pos)

    def _decode(self, emb: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Teacher-forced or single-step decode.
        emb:    (B, T, D)
        memory: (B, 1, D)
        Returns logits (B, T, full_vocab)
        """
        T = emb.size(1)
        causal = nn.Transformer.generate_square_subsequent_mask(T, device=emb.device)
        out = self.decoder(emb, memory, tgt_mask=causal)
        return self.out_proj(out)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        context: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            context:    (B, D) fused multimodal embedding
            target_ids: (B, T) ground-truth token indices for teacher forcing.
                        If None, runs greedy generation and returns token ids.
        Returns:
            If target_ids given: logits (B, T, vocab_size+2)
            Otherwise:           token ids (B, max_len) from greedy decode
        """
        memory = self.context_proj(context).unsqueeze(1)  # (B, 1, D)

        if target_ids is not None:
            B, T = target_ids.shape
            bos_col = torch.full((B, 1), self.BOS, dtype=torch.long, device=context.device)
            dec_in = torch.cat([bos_col, target_ids[:, :-1]], dim=1)  # right-shift
            return self._decode(self._embed(dec_in), memory)           # (B, T, V)

        return self.generate(context, method="greedy")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        context: torch.Tensor,
        method: str = "greedy",
        beam_size: int = 3,
    ) -> torch.Tensor:
        """
        Args:
            context:   (B, D)
            method:    "greedy" or "beam"
            beam_size: number of beams (only used when method=="beam")
        Returns:
            token_ids: (B, ≤max_len) generated answer tokens (BOS/EOS stripped)
        """
        memory = self.context_proj(context).unsqueeze(1)  # (B, 1, D)
        if method == "beam" and beam_size > 1:
            return self._beam_decode(memory, beam_size)
        return self._greedy_decode(memory)

    def _greedy_decode(self, memory: torch.Tensor) -> torch.Tensor:
        B = memory.size(0)
        device = memory.device
        tokens = torch.full((B, 1), self.BOS, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(self.max_len):
            logits = self._decode(self._embed(tokens), memory)  # (B, T, V)
            next_tok = logits[:, -1].argmax(dim=-1)             # (B,)
            finished |= next_tok == self.EOS
            tokens = torch.cat([tokens, next_tok.unsqueeze(1)], dim=1)
            if finished.all():
                break

        return tokens[:, 1:]  # strip BOS

    def _beam_decode(self, memory: torch.Tensor, beam_size: int) -> torch.Tensor:
        B = memory.size(0)
        device = memory.device
        full_vocab = self.vocab_size + 2

        # Expand memory: (B*beam, 1, D)
        mem_exp = memory.repeat_interleave(beam_size, dim=0)

        # tokens: (B, beam, T)
        tokens = torch.full((B, beam_size, 1), self.BOS, dtype=torch.long, device=device)
        # log-scores: (B, beam) — only first beam per sample is alive at start
        log_scores = torch.full((B, beam_size), float("-inf"), device=device)
        log_scores[:, 0] = 0.0

        for _ in range(self.max_len):
            T = tokens.size(2)
            # Flatten beams for decoding: (B*beam, T)
            tok_flat = tokens.view(B * beam_size, T)
            logits = self._decode(self._embed(tok_flat), mem_exp)   # (B*beam, T, V)
            lp = F.log_softmax(logits[:, -1], dim=-1)               # (B*beam, V)
            lp = lp.view(B, beam_size, full_vocab)

            # Total scores: (B, beam, V)
            total = log_scores.unsqueeze(-1) + lp
            total_flat = total.view(B, beam_size * full_vocab)

            # Select top-beam_size continuations
            top_scores, top_idx = total_flat.topk(beam_size, dim=-1)   # (B, beam)
            from_beam = top_idx // full_vocab                           # (B, beam)
            next_tok = top_idx % full_vocab                             # (B, beam)

            # Gather correct previous token sequences
            b_idx = torch.arange(B, device=device).unsqueeze(-1).expand_as(from_beam)
            prev_tokens = tokens[b_idx, from_beam]                     # (B, beam, T)
            tokens = torch.cat([prev_tokens, next_tok.unsqueeze(-1)], dim=-1)
            log_scores = top_scores

        # Return best beam (highest log-score), strip BOS
        best = log_scores.argmax(dim=-1)                               # (B,)
        b_idx = torch.arange(B, device=device)
        return tokens[b_idx, best, 1:]                                 # (B, T)
