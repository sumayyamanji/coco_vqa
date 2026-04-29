"""RAG-style retriever: augments a question with retrieved visual context."""
from __future__ import annotations

from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from .embedder import CLIPEmbedder
    from .index import VQAIndex


class RAGRetriever:
    """Retrieves visually similar training examples and formats them as context.

    Typical usage
    -------------
    Load once at the start of inference::

        embedder = CLIPEmbedder()
        index    = VQAIndex.load(cfg["retrieval"]["faiss_index_path"])
        retriever = RAGRetriever(index, embedder)

    Then for each example before tokenisation::

        augmented_question = retriever.augment_question(pil_image, question, k=3)
        # Pass augmented_question to the BERT tokeniser instead of the raw question.

    Parameters
    ----------
    index:    Pre-built VQAIndex
    embedder: CLIPEmbedder (same model used to build the index)
    """

    def __init__(self, index: "VQAIndex", embedder: "CLIPEmbedder") -> None:
        self.index = index
        self.embedder = embedder

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve_for_image(
        self,
        pil_image: Image.Image,
        question: str,
        k: int = 3,
    ) -> str:
        """Retrieve top-k similar images and return a context string.

        The context string summarises retrieved examples in natural language::

            "Similar images show: <caption>. Related Q: <q> A: <a>. ..."

        If a retrieved entry has no caption, that sentence is omitted.

        Parameters
        ----------
        pil_image: query image (PIL.Image)
        question:  raw question string (used only to fall back on text if
                   the image embed fails; the retrieved entries come from
                   the visual search)
        k:         number of neighbours to retrieve

        Returns
        -------
        Context string (may be empty if the index returns no results).
        """
        query_vec = self.embedder.embed_image(pil_image)
        hits = self.index.search(query_vec, k=k)
        return _format_context(hits)

    def augment_question(
        self,
        pil_image: Image.Image,
        question: str,
        k: int = 3,
    ) -> str:
        """Return the question prepended with retrieved visual context.

        The returned string is ready to be passed straight to the BERT
        tokeniser in place of the bare question::

            f"{context} Question: {question}"

        If retrieval returns nothing, the bare question is returned unchanged.

        Parameters
        ----------
        pil_image: query image
        question:  original question text
        k:         neighbours to retrieve
        """
        context = self.retrieve_for_image(pil_image, question, k=k)
        if not context:
            return question
        return f"{context} Question: {question}"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _format_context(hits: list) -> str:
    """Convert a list of search result dicts into a human-readable string.

    Each hit contributes at most two sentences:
    * "Similar images show: <caption>."   — only when caption is non-empty
    * "Related Q: <question> A: <answer>."
    """
    parts: list[str] = []
    for hit in hits:
        caption: str = (hit.get("caption") or "").strip()
        q: str = (hit.get("question") or "").strip()
        a: str = (hit.get("answer") or "").strip()

        if caption:
            parts.append(f"Similar images show: {caption}.")
        if q and a:
            parts.append(f"Related Q: {q} A: {a}.")
        elif q:
            parts.append(f"Related Q: {q}.")

    return " ".join(parts)
