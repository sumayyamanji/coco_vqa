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
