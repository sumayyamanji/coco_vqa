"""CLIP-based image and text embedder for RAG retrieval."""
from __future__ import annotations

from typing import List

import numpy as np
import torch
from PIL import Image


class CLIPEmbedder:
    """Wraps openai/clip-vit-large-patch14 to produce L2-normalised embeddings.

    All outputs are float32 numpy arrays suitable for use with a FAISS
    IndexFlatIP (inner product == cosine similarity on unit vectors).

    Parameters
    ----------
    model_name: HuggingFace model ID (defaults to CLIP ViT-L/14)
    device:     "cuda", "cpu", or None (auto-detects)
    batch_size: chunk size used internally by the batch methods
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        device: str | None = None,
        batch_size: int = 64,
    ) -> None:
        from transformers import CLIPModel, CLIPProcessor

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.batch_size = batch_size
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Single-sample helpers
    # ------------------------------------------------------------------

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """Embed one PIL image → L2-normalised float32 numpy vector."""
        return self.embed_images([image])[0]

    def embed_text(self, text: str) -> np.ndarray:
        """Embed one text string → L2-normalised float32 numpy vector."""
        return self.embed_texts([text])[0]

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def embed_images(self, images: List[Image.Image]) -> np.ndarray:
        """Embed a list of PIL images.

        Returns
        -------
        np.ndarray of shape (N, D), float32, each row L2-normalised.
        """
        all_vecs: List[np.ndarray] = []
        for start in range(0, len(images), self.batch_size):
            chunk = images[start : start + self.batch_size]
            inputs = self.processor(images=chunk, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                feats = self.model.get_image_features(**inputs)
            vecs = feats.cpu().float().numpy()
            all_vecs.append(_l2_normalise(vecs))
        return np.concatenate(all_vecs, axis=0) if all_vecs else np.empty((0, 0), dtype=np.float32)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of strings.

        Returns
        -------
        np.ndarray of shape (N, D), float32, each row L2-normalised.
        """
        all_vecs: List[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start : start + self.batch_size]
            inputs = self.processor(
                text=chunk, return_tensors="pt", padding=True, truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                feats = self.model.get_text_features(**inputs)
            vecs = feats.cpu().float().numpy()
            all_vecs.append(_l2_normalise(vecs))
        return np.concatenate(all_vecs, axis=0) if all_vecs else np.empty((0, 0), dtype=np.float32)


# ---------------------------------------------------------------------------
# Internal utility
# ---------------------------------------------------------------------------

def _l2_normalise(vecs: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation; zero-vectors are left as-is."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (vecs / norms).astype(np.float32)
