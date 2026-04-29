"""RAG-style retrieval module for VQA context augmentation."""
from .embedder import CLIPEmbedder
from .index import VQAIndex
from .retriever import RAGRetriever

__all__ = ["CLIPEmbedder", "VQAIndex", "RAGRetriever"]
