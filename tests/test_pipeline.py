"""Smoke tests for the full VQA pipeline (no GPU required)."""
from __future__ import annotations

import torch
import pytest
from unittest.mock import MagicMock, patch


# ------------------------------------------------------------------ helpers --

def _dummy_vocab(n: int = 10):
    from src.data.answer_vocab import AnswerVocab
    v = AnswerVocab(["yes", "no", "cat", "dog", "red", "blue", "one", "two", "three", "many"])
    return v


def _dummy_batch(B: int = 2, seq_len: int = 10, img_size: int = 224):
    return {
        "image": torch.randn(B, 3, img_size, img_size),
        "input_ids": torch.randint(0, 1000, (B, seq_len)),
        "attention_mask": torch.ones(B, seq_len, dtype=torch.long),
        "label": torch.randint(0, 10, (B,)),
        "soft_scores": torch.rand(B, 10),
        "question_id": list(range(B)),
    }


# ------------------------------------------------------------------- tests --

class TestAnswerVocab:
    def test_roundtrip(self, tmp_path):
        from src.data.answer_vocab import AnswerVocab
        vocab = AnswerVocab(["yes", "no", "cat"])
        path = tmp_path / "vocab.json"
        vocab.save(path)
        loaded = AnswerVocab.load(path)
        assert len(loaded) == len(vocab)
        assert loaded.answer_to_idx("yes") == vocab.answer_to_idx("yes")

    def test_soft_scores_sum_to_at_most_one_per_class(self):
        vocab = _dummy_vocab()
        answers = [{"answer": "yes"}, {"answer": "yes"}, {"answer": "yes"}, {"answer": "no"}]
        scores = vocab.build_soft_scores(answers)
        assert scores.max() <= 1.0
        assert scores[vocab.answer_to_idx("yes")] == pytest.approx(1.0)  # 3/3

    def test_unk_handling(self):
        vocab = _dummy_vocab()
        idx = vocab.answer_to_idx("xyzzy_not_in_vocab")
        assert idx == vocab.answer_to_idx(vocab.UNK)


class TestAugmentations:
    def test_train_transforms_shape(self):
        from src.data.augmentations import get_train_transforms
        from PIL import Image
        import numpy as np
        t = get_train_transforms(224)
        img = Image.fromarray(np.random.randint(0, 255, (300, 400, 3), dtype="uint8"))
        out = t(img)
        assert out.shape == (3, 224, 224)

    def test_val_transforms_deterministic(self):
        from src.data.augmentations import get_val_transforms
        from PIL import Image
        import numpy as np
        t = get_val_transforms(224)
        img = Image.fromarray(np.random.randint(0, 255, (300, 400, 3), dtype="uint8"))
        out1, out2 = t(img), t(img)
        assert torch.allclose(out1, out2)


class TestLoss:
    def test_vqa_loss_positive(self):
        from src.training.losses import VQALoss
        loss_fn = VQALoss(label_smoothing=0.0)
        logits = torch.randn(4, 10)
        scores = torch.rand(4, 10)
        loss = loss_fn(logits, scores)
        assert loss.item() > 0

    def test_label_smoothing_increases_loss_on_confident_prediction(self):
        from src.training.losses import VQALoss
        logits = torch.zeros(2, 10)
        logits[:, 0] = 10.0  # very confident
        scores = torch.zeros(2, 10)
        scores[:, 0] = 1.0
        loss_smooth = VQALoss(label_smoothing=0.1)(logits, scores)
        loss_hard = VQALoss(label_smoothing=0.0)(logits, scores)
        assert loss_smooth > loss_hard


class TestScheduler:
    def test_warmup_monotone_increase(self):
        from src.training.scheduler import build_scheduler
        model = torch.nn.Linear(4, 4)
        opt = torch.optim.SGD(model.parameters(), lr=1e-3)
        sched = build_scheduler(opt, warmup_steps=10, total_steps=100)
        lrs = []
        for _ in range(10):
            sched.step()
            lrs.append(opt.param_groups[0]["lr"])
        assert all(lrs[i] <= lrs[i + 1] for i in range(len(lrs) - 1))


class TestMetrics:
    def test_soft_accuracy_perfect(self):
        from src.evaluation.metrics import soft_accuracy
        logits = torch.zeros(3, 5)
        logits[0, 2] = 10; logits[1, 0] = 10; logits[2, 4] = 10
        scores = torch.zeros(3, 5)
        scores[0, 2] = 1.0; scores[1, 0] = 1.0; scores[2, 4] = 1.0
        assert soft_accuracy(logits, scores) == pytest.approx(1.0)

    def test_top_k_accuracy_ge_top1(self):
        from src.evaluation.metrics import soft_accuracy, top_k_accuracy
        logits = torch.randn(8, 20)
        scores = torch.rand(8, 20)
        assert top_k_accuracy(logits, scores, k=3) >= soft_accuracy(logits, scores) - 1e-6


class TestCheckpoint:
    def test_save_load_roundtrip(self, tmp_path):
        from src.utils.checkpoint import save_checkpoint, load_checkpoint
        model = torch.nn.Linear(4, 4)
        opt = torch.optim.Adam(model.parameters())
        save_checkpoint(model, opt, epoch=5, ckpt_dir=tmp_path, keep_last_n=3)
        ckpt_files = list(tmp_path.glob("*.pt"))
        assert len(ckpt_files) == 1
        model2 = torch.nn.Linear(4, 4)
        epoch = load_checkpoint(ckpt_files[0], model2)
        assert epoch == 5
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)

    def test_keep_last_n_rotation(self, tmp_path):
        from src.utils.checkpoint import save_checkpoint
        model = torch.nn.Linear(4, 4)
        opt = torch.optim.Adam(model.parameters())
        for epoch in range(1, 6):
            save_checkpoint(model, opt, epoch=epoch, ckpt_dir=tmp_path, keep_last_n=3)
        assert len(list(tmp_path.glob("*.pt"))) == 3


class TestNewLosses:
    def test_vqa_soft_loss_positive(self):
        from src.training.losses import VQASoftLoss
        cfg = {"training": {"label_smoothing": 0.1}}
        loss_fn = VQASoftLoss(cfg)
        logits = torch.randn(4, 10)
        scores = torch.rand(4, 10)
        loss = loss_fn(logits, scores)
        assert loss.item() > 0

    def test_answer_type_loss_correct_class(self):
        from src.training.losses import AnswerTypeLoss
        loss_fn = AnswerTypeLoss()
        logits = torch.zeros(3, 3)
        logits[0, 0] = 10.0; logits[1, 1] = 10.0; logits[2, 2] = 10.0
        labels = torch.tensor([0, 1, 2])
        loss = loss_fn(logits, labels)
        assert loss.item() < 0.01

    def test_total_loss_returns_components(self):
        from src.training.losses import TotalLoss
        cfg = {"training": {"label_smoothing": 0.0}}
        loss_fn = TotalLoss(cfg)
        answer_logits = torch.randn(2, 10)
        type_logits = torch.randn(2, 3)
        answer_scores = torch.rand(2, 10)
        type_labels = torch.randint(0, 3, (2,))
        total, breakdown = loss_fn(answer_logits, type_logits, answer_scores, type_labels)
        assert total.item() > 0
        assert "vqa_loss" in breakdown and "type_loss" in breakdown


class TestAdditionalMetrics:
    def test_per_type_accuracy_keys(self):
        from src.evaluation.metrics import per_type_accuracy
        logits = torch.randn(6, 5)
        scores = torch.rand(6, 5)
        types = ["yes/no", "number", "other", "yes/no", "other", "number"]
        result = per_type_accuracy(logits, scores, types)
        assert set(result.keys()) == {"yes/no", "number", "other"}

    def test_confusion_matrix_shape_and_diagonal(self):
        from src.evaluation.metrics import compute_confusion_matrix
        y_true = ["yes/no", "yes/no", "number", "other"]
        y_pred = [0, 0, 1, 2]
        cm = compute_confusion_matrix(y_true, y_pred)
        assert cm.shape == (3, 3)
        assert cm[0, 0] == 2

    def test_bias_analysis_counts_sum_correctly(self):
        from src.evaluation.metrics import bias_analysis
        text_only = {1: {"score": 1.0}, 2: {"score": 0.0}, 3: {"score": 1.0}}
        multimodal = {1: {"score": 0.0}, 2: {"score": 1.0}, 3: {"score": 1.0}}
        annotations = {"annotations": [
            {"question_id": 1}, {"question_id": 2}, {"question_id": 3}
        ]}
        result = bias_analysis(text_only, multimodal, annotations)
        total = (result["language_bias_count"] + result["multimodal_gain_count"]
                 + result["both_correct_count"] + result["both_fail_count"])
        assert total == 3


class TestVQAModelMocked:
    def _make_model(self, vocab_size: int = 10):
        """Build VQAModel with real tiny linear layers (no pretrained weights)."""
        import torch.nn as nn
        from src.models.vqa_model import VQAModel
        cfg = {
            "model": {
                "vision_backbone": "openai/clip-vit-large-patch14",
                "text_encoder": "bert-base-uncased",
                "hidden_dim": 32,
                "num_heads": 4,
                "fusion_layers": 1,
                "num_answer_classes": vocab_size,
                "dropout": 0.0,
                "fusion_type": "cross_attention",
                "use_scene_graph": False,
            },
            "training": {"label_smoothing": 0.0},
        }
        with patch("src.models.vision_encoder.CLIPVisionEncoder.__init__", return_value=None), \
             patch("src.models.text_encoder.BERTTextEncoder.__init__", return_value=None):
            model = VQAModel.__new__(VQAModel)
            model.config = cfg
            model.mode = "multimodal"
            model.vocab_size = vocab_size
            model.scene_graph = None
            model._fusion_type = "bilinear"
            from src.models.fusion import BilinearFusion
            from src.models.answer_heads import (
                AnswerTypeClassifier, YesNoHead, NumberHead, OpenEndedHead, GenerativeHead
            )
            model.fusion = BilinearFusion(cfg)
            model.answer_type_clf = AnswerTypeClassifier(32, 0.0)
            model.yes_no_head = YesNoHead(32)
            model.number_head = NumberHead(32)
            model.open_ended_head = OpenEndedHead(32, vocab_size, 0.0)
            model.generative_head = GenerativeHead(32, vocab_size, max_len=5,
                                                    num_layers=1, num_heads=4, dropout=0.0)
        return model

    def test_answer_type_clf_output_shape(self):
        from src.models.answer_heads import AnswerTypeClassifier
        clf = AnswerTypeClassifier(hidden_dim=32, dropout=0.0)
        x = torch.randn(3, 32)
        out = clf(x)
        assert out.shape == (3, 3)

    def test_open_ended_head_output_shape(self):
        from src.models.answer_heads import OpenEndedHead
        head = OpenEndedHead(hidden_dim=32, num_classes=10, dropout=0.0)
        x = torch.randn(2, 32)
        out = head(x)
        assert out.shape == (2, 10)

    def test_bilinear_fusion_output_shape(self):
        from src.models.fusion import BilinearFusion
        cfg = {"model": {"hidden_dim": 32, "dropout": 0.0, "num_heads": 4,
                         "fusion_layers": 1, "num_answer_classes": 10}}
        fusion = BilinearFusion(cfg)
        vis = torch.randn(2, 32)
        txt = torch.randn(2, 32)
        out = fusion(vis, txt)
        assert out.shape == (2, 32)


class TestGradCAMHooks:
    def test_hooks_registered_on_module(self):
        from src.utils.gradcam import GradCAM
        model = torch.nn.Sequential(torch.nn.Linear(4, 4))
        gcam = GradCAM(model, model[0])
        assert len(gcam._hooks) == 2
        gcam.remove_hooks()
        assert len(gcam._hooks) == 0

    def test_string_layer_lookup_raises_on_unknown(self):
        from src.utils.gradcam import GradCAM
        model = torch.nn.Linear(4, 4)
        with pytest.raises(ValueError, match="not found"):
            GradCAM(model, "nonexistent.layer.name")

    def test_overlay_returns_pil_image(self):
        from src.utils.gradcam import GradCAM
        from PIL import Image
        import numpy as np
        model = torch.nn.Linear(4, 4)
        gcam = GradCAM(model, model)
        gcam.remove_hooks()
        pil = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype="uint8"))
        heatmap = np.random.rand(32, 32).astype("float32")
        result = gcam.overlay(pil, heatmap, alpha=0.4)
        assert isinstance(result, Image.Image)
        assert result.size == (32, 32)
