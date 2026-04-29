"""Model components and assembled VQA model."""
from .vqa_model import VQAModel
from .vision_encoder import CLIPVisionEncoder, VisionEncoder
from .text_encoder import BERTTextEncoder, TextEncoder
from .fusion import CrossModalFusion, BilinearFusion, CrossAttentionFusion, CrossModalBlock
from .answer_heads import (
    AnswerClassifier,
    YesNoHead,
    NumberHead,
    OpenEndedHead,
    AnswerTypeClassifier,
    GenerativeHead,
)
from .scene_graph import SceneGraphGenerator, SceneGraphEncoder, GCNLayer

__all__ = [
    "VQAModel",
    # Encoders
    "CLIPVisionEncoder",
    "VisionEncoder",
    "BERTTextEncoder",
    "TextEncoder",
    # Fusion
    "CrossModalFusion",
    "CrossModalBlock",
    "BilinearFusion",
    "CrossAttentionFusion",
    # Answer heads
    "AnswerClassifier",
    "YesNoHead",
    "NumberHead",
    "OpenEndedHead",
    "AnswerTypeClassifier",
    "GenerativeHead",
    # Scene graph
    "SceneGraphGenerator",
    "SceneGraphEncoder",
    "GCNLayer",
]
