"""FAISS-backed index of VQA training examples for RAG retrieval."""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

_log = logging.getLogger(__name__)


class VQAIndex:
    """FAISS IndexFlatIP over CLIP image embeddings of VQA training examples.

    Build once, save to disk, then load in future runs.

    File layout under save_path/
    ----------------------------
    index.faiss   — FAISS binary index
    metadata.json — list of dicts {image_id, question, answer, caption}

    Usage
    -----
    Build::

        embedder = CLIPEmbedder()
        idx = VQAIndex.build_from_dataset(train_dataset, embedder, "data/faiss_index")

    Load and search::

        idx = VQAIndex.load("data/faiss_index")
        results = idx.search(query_vec, k=5)
    """

    def __init__(self, faiss_index, metadata: List[Dict[str, Any]]) -> None:
        self._index = faiss_index
        self._metadata = metadata

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    @classmethod
    def build_from_dataset(
        cls,
        dataset,
        embedder,
        save_path: str | Path,
        batch_size: int = 64,
        captions: Optional[Dict[int, str]] = None,
    ) -> "VQAIndex":
        """Embed all training images and build an IndexFlatIP.

        Parameters
        ----------
        dataset:    VQADataset instance (needs .samples, .images_dir, ._img_prefix)
        embedder:   CLIPEmbedder instance
        save_path:  directory to write index.faiss + metadata.json
        batch_size: images to embed per forward pass
        captions:   optional {image_id: caption_string} map; defaults to ""
        """
        import faiss  # type: ignore

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        samples = dataset.samples
        images_dir = dataset.images_dir
        img_prefix = dataset._img_prefix
        captions = captions or {}

        all_vecs: List[np.ndarray] = []
        metadata: List[Dict[str, Any]] = []
        faiss_index = None

        _log.info("Building FAISS index over %d samples …", len(samples))

        for start in range(0, len(samples), batch_size):
            chunk = samples[start : start + batch_size]
            pil_images: List[Image.Image] = []
            chunk_meta: List[Dict[str, Any]] = []

            for sample in chunk:
                image_id: int = sample["image_id"]
                img_path = images_dir / f"{img_prefix}_{image_id:012d}.jpg"

                if img_path.exists():
                    try:
                        pil_img = Image.open(img_path).convert("RGB")
                    except Exception:
                        _log.warning("Could not open %s, skipping.", img_path)
                        continue
                else:
                    _log.warning("Image not found: %s, skipping.", img_path)
                    continue

                top_answer = _top_answer(sample.get("answers", []))
                pil_images.append(pil_img)
                chunk_meta.append(
                    {
                        "image_id": image_id,
                        "question": sample.get("question", ""),
                        "answer": top_answer,
                        "caption": captions.get(image_id, ""),
                    }
                )

            if not pil_images:
                continue

            vecs = embedder.embed_images(pil_images)  # (chunk, D)

            if faiss_index is None:
                dim = vecs.shape[1]
                faiss_index = faiss.IndexFlatIP(dim)
                _log.info("Created IndexFlatIP with dim=%d", dim)

            faiss_index.add(vecs)
            all_vecs.append(vecs)
            metadata.extend(chunk_meta)

            if (start // batch_size) % 10 == 0:
                _log.info("  indexed %d / %d", len(metadata), len(samples))

        if faiss_index is None:
            raise RuntimeError("No images could be embedded — check your dataset paths.")

        _log.info("Index built: %d vectors.", faiss_index.ntotal)

        # Persist
        index_path = save_path / "index.faiss"
        meta_path = save_path / "metadata.json"
        faiss.write_index(faiss_index, str(index_path))
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
        _log.info("Saved index → %s, metadata → %s", index_path, meta_path)

        return cls(faiss_index, metadata)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> "VQAIndex":
        """Load a pre-built index from disk.

        Parameters
        ----------
        path: directory containing index.faiss + metadata.json
        """
        import faiss  # type: ignore

        path = Path(path)
        index_path = path / "index.faiss"
        meta_path = path / "metadata.json"

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found at {meta_path}")

        faiss_index = faiss.read_index(str(index_path))
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        _log.info("Loaded index with %d vectors from %s", faiss_index.ntotal, path)
        return cls(faiss_index, metadata)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self, query_embedding: np.ndarray, k: int = 5
    ) -> List[Dict[str, Any]]:
        """Return the top-k most similar entries.

        Parameters
        ----------
        query_embedding: 1-D float32 numpy array, L2-normalised
        k:               number of results

        Returns
        -------
        list of dicts — each is the metadata entry with an added "score" key
        (cosine similarity in [-1, 1], higher is better)
        """
        query = query_embedding.astype(np.float32)
        if query.ndim == 1:
            query = query[np.newaxis, :]  # (1, D)

        scores, indices = self._index.search(query, k)  # both (1, k)
        scores = scores[0]
        indices = indices[0]

        results = []
        for score, idx in zip(scores.tolist(), indices.tolist()):
            if idx < 0 or idx >= len(self._metadata):
                continue
            entry = dict(self._metadata[idx])
            entry["score"] = float(score)
            results.append(entry)

        return results

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return self._index.ntotal


# ---------------------------------------------------------------------------
# Internal utility
# ---------------------------------------------------------------------------

def _top_answer(answers: List[Dict[str, Any]]) -> str:
    """Return the single most frequent answer string."""
    if not answers:
        return ""
    counts = Counter(a.get("answer", "") for a in answers)
    return counts.most_common(1)[0][0]
