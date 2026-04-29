"""VQAModel — assembles all components into a single end-to-end nn.Module."""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vision_encoder import CLIPVisionEncoder
from .text_encoder import BERTTextEncoder
from .fusion import CrossModalFusion, BilinearFusion
from .answer_heads import (
    AnswerTypeClassifier,
    YesNoHead,
    NumberHead,
    OpenEndedHead,
    GenerativeHead,
)
from .scene_graph import SceneGraphGenerator


class VQAModel(nn.Module):
    """End-to-end VQA model that reads all hyper-parameters from a config dict.

    Architecture (multimodal mode)
    ------------------------------
    1. CLIPVisionEncoder  → patch_embeddings (B, N, D)  + vis_cls (B, D)
    2. BERTTextEncoder    → token_embeddings (B, T, D)  + text_cls (B, D)
    3. SceneGraphGenerator (optional enrichment of patch embeddings)
    4. CrossModalFusion   — bidirectional cross-modal attention; OR
       BilinearFusion     — Tucker-style pooling of the two CLS vectors
    5. AnswerTypeClassifier → (B, 3)   yes/no / number / other
    6. YesNoHead          → (B, 2)    auxiliary
       NumberHead         → (B, 50)   auxiliary
       OpenEndedHead      → (B, C)    primary logits used for training loss
    7. GenerativeHead     — optional free-form decoder (call generate_answer)

    The ``answer_logits`` in the forward output always come from OpenEndedHead
    (the full-vocab primary head). The specialised heads provide auxiliary
    predictions that can be used for multi-task losses or type-specific analysis.

    Supported modes
    ---------------
    multimodal  — full pipeline (default)
    text_only   — skip vision encoder and fusion; use BERT CLS directly
    image_only  — skip text encoder and fusion; use CLIP CLS directly

    Args:
        config:     Parsed config.yaml dict
        vocab_size: Number of answer classes (typically 3129)
        mode:       "multimodal" | "text_only" | "image_only"
    """

    _ANSWER_TYPE_MAP = {"yes/no": 0, "number": 1, "other": 2}

    def __init__(
        self,
        config: Dict[str, Any],
        vocab_size: int,
        mode: str = "multimodal",
    ) -> None:
        super().__init__()
        self.config = config
        self.mode = mode
        self.vocab_size = vocab_size

        m = config["model"]
        hidden_dim: int = m["hidden_dim"]
        dropout: float = m.get("dropout", 0.1)
        fusion_type: str = m.get("fusion_type", "cross_attention")
        use_scene_graph: bool = m.get("use_scene_graph", False)

        # ---- Encoders ----
        self.vision_encoder = CLIPVisionEncoder(config)
        self.text_encoder = BERTTextEncoder(config)

        # ---- Optional scene graph ----
        self.scene_graph: Optional[SceneGraphGenerator] = (
            SceneGraphGenerator(config) if use_scene_graph and mode == "multimodal" else None
        )

        # ---- Fusion ----
        if fusion_type == "bilinear":
            self.fusion = BilinearFusion(config)
            self._fusion_type = "bilinear"
        else:
            self.fusion = CrossModalFusion(config)
            self._fusion_type = "cross_attention"

        # ---- Answer heads ----
        self.answer_type_clf = AnswerTypeClassifier(hidden_dim, dropout)
        self.yes_no_head = YesNoHead(hidden_dim)
        self.number_head = NumberHead(hidden_dim)
        self.open_ended_head = OpenEndedHead(hidden_dim, vocab_size, dropout)

        # ---- Generative head ----
        self.generative_head = GenerativeHead(
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            max_len=config.get("generation", {}).get("max_len", 10),
            num_layers=4,
            num_heads=m.get("num_heads", 8),
            dropout=dropout,
        )

    # ------------------------------------------------------------------
    # Internal: encode + fuse
    # ------------------------------------------------------------------

    def _encode_and_fuse(
        self, batch: Dict[str, Any]
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Run encoders and fusion, respecting the operating mode."""
        pixel_values: Optional[torch.Tensor] = batch.get("image_tensor")
        input_ids: Optional[torch.Tensor] = batch.get("question_ids")
        attention_mask: Optional[torch.Tensor] = batch.get("attention_mask")
        object_labels: Optional[torch.Tensor] = batch.get("object_labels")

        patch_emb: Optional[torch.Tensor] = None
        vis_cls: Optional[torch.Tensor] = None
        tok_emb: Optional[torch.Tensor] = None
        text_cls: Optional[torch.Tensor] = None
        attn_weights: Optional[torch.Tensor] = None
        sg_out: Optional[torch.Tensor] = None

        if self.mode != "text_only" and pixel_values is not None:
            patch_emb, vis_cls = self.vision_encoder(pixel_values)

        if self.mode != "image_only" and input_ids is not None:
            tok_emb, text_cls = self.text_encoder(input_ids, attention_mask)

        # Optional scene graph enrichment of patch embeddings
        if self.scene_graph is not None and patch_emb is not None:
            sg_out = self.scene_graph(patch_emb, object_labels)
            patch_emb = sg_out  # use enriched patches for fusion

        # Fuse
        if self.mode == "multimodal" and patch_emb is not None and tok_emb is not None:
            key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
            if self._fusion_type == "bilinear":
                # BilinearFusion takes two global vectors
                fused = self.fusion(vis_cls, text_cls)
                attn_weights = None
            else:
                fused, tok_emb, patch_emb, attn_weights = self.fusion(
                    tok_emb, patch_emb, key_padding_mask
                )
        elif self.mode == "text_only":
            fused = text_cls if text_cls is not None else torch.zeros(
                1, self.config["model"]["hidden_dim"],
                device=next(self.parameters()).device,
            )
        else:  # image_only
            fused = vis_cls if vis_cls is not None else torch.zeros(
                1, self.config["model"]["hidden_dim"],
                device=next(self.parameters()).device,
            )

        return {
            "fused": fused,
            "patch_embeddings": patch_emb,
            "token_embeddings": tok_emb,
            "cross_attention_weights": attn_weights,
            "scene_graph_output": sg_out,
        }

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Full forward pass.

        Args:
            batch: dict from VQADataset / DataLoader with keys:
                   image_tensor, question_ids, attention_mask,
                   answer_scores, answer_type, question_id, …

        Returns dict with:
            answer_type_logits    (B, 3)
            answer_logits         (B, vocab_size)  — from OpenEndedHead
            yes_no_logits         (B, 2)
            number_logits         (B, 50)
            top3_answers          dict(indices=(B,3), probs=(B,3))
            confidence            (B,)  max softmax probability
            patch_embeddings      (B, N, D) or None
            cross_attention_weights (B, T, N) or None
            scene_graph_output    (B, N, D) or None
        """
        enc = self._encode_and_fuse(batch)
        fused: torch.Tensor = enc["fused"]

        # ---- Classification heads ----
        type_logits = self.answer_type_clf(fused)   # (B, 3)
        yes_no_logits = self.yes_no_head(fused)     # (B, 2)
        number_logits = self.number_head(fused)     # (B, 50)
        answer_logits = self.open_ended_head(fused) # (B, vocab_size)

        # ---- Top-3 answers + confidence ----
        probs = F.softmax(answer_logits, dim=-1)                       # (B, C)
        top3 = probs.topk(3, dim=-1)                                   # values, indices
        confidence = probs.max(dim=-1).values                          # (B,)

        return {
            "answer_type_logits": type_logits,
            "answer_logits": answer_logits,
            "yes_no_logits": yes_no_logits,
            "number_logits": number_logits,
            "top3_answers": {"indices": top3.indices, "probs": top3.values},
            "confidence": confidence,
            "patch_embeddings": enc["patch_embeddings"],
            "cross_attention_weights": enc["cross_attention_weights"],
            "scene_graph_output": enc["scene_graph_output"],
        }

    # ------------------------------------------------------------------
    # Generative interface
    # ------------------------------------------------------------------

    def generate_answer(
        self,
        batch: Dict[str, Any],
        method: str = "greedy",
        beam_size: int = 3,
    ) -> torch.Tensor:
        """Generate answer token ids using the autoregressive GenerativeHead.

        Args:
            batch:     same batch dict as forward()
            method:    "greedy" or "beam"
            beam_size: number of beams (only used when method=="beam")
        Returns:
            token_ids: (B, ≤max_len) generated answer token indices into the
                       answer vocab (vocab_size), with BOS/EOS stripped.
        """
        enc = self._encode_and_fuse(batch)
        return self.generative_head.generate(
            enc["fused"], method=method, beam_size=beam_size
        )

    # ------------------------------------------------------------------
    # Convenience: type-routed answer at inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self, batch: Dict[str, Any], top_k: int = 3
    ) -> Dict[str, Any]:
        """Like forward() but returns a human-readable summary dict.

        Routes the final prediction through the type-predicted head:
          yes/no → YesNoHead argmax  (0=no, 1=yes)
          number → NumberHead argmax (interpreted as 0-49)
          other  → OpenEndedHead top-k
        """
        out = self.forward(batch)

        predicted_type = out["answer_type_logits"].argmax(dim=-1)  # (B,)
        probs = F.softmax(out["answer_logits"], dim=-1)
        topk = probs.topk(top_k, dim=-1)

        return {
            "predicted_type": predicted_type,
            "top_k_indices": topk.indices,
            "top_k_probs": topk.values,
            "confidence": out["confidence"],
            "yes_no_pred": out["yes_no_logits"].argmax(dim=-1),
            "number_pred": out["number_logits"].argmax(dim=-1),
        }
