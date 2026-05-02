

import sys
import json
import random
from pathlib import Path
from collections import Counter

import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import ROOT_DIR, setup_output_dirs
from src.data.answer_vocab import load_vocab
from src.data.dataset import VQADataset
from src.data.augmentations import get_val_transforms
from src.models.vqa_model import VQAModel

def main():
    # Load config
    with open(ROOT_DIR / "configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"Device: {device}")

    # Load vocab
    vocab_path = ROOT_DIR / cfg["paths"]["vocab_path"]
    ans2idx, idx2ans = load_vocab(vocab_path)
    print(f"Vocab size: {len(ans2idx)}")

    # Load 100 random val samples
    transform = get_val_transforms(cfg["data"]["image_size"])
    dataset = VQADataset(
        split="val",
        config=cfg,
        vocab=ans2idx,
        transform=transform,
        mode="multimodal"
    )

    # Sample 100 random indices
    random.seed(42)
    indices = random.sample(range(len(dataset)), 100)

    # Load model
    checkpoint_path = ROOT_DIR / "outputs/checkpoints/best_model.pt"
    if not checkpoint_path.exists():
        print("No checkpoint found at outputs/checkpoints/best_model.pt")
        return

    model = VQAModel(
        config=cfg,
        vocab_size=len(ans2idx),
        mode="multimodal"
    )

    checkpoint = torch.load(checkpoint_path,
                            map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    best_acc = checkpoint.get('best_val_accuracy', '?')
    acc_str = f"{best_acc:.4f}" if isinstance(best_acc, (int, float)) else str(best_acc)
    print(f"Loaded checkpoint from epoch "
          f"{checkpoint.get('epoch', '?')}, "
          f"best acc: {acc_str}")

    # Run inference
    predictions = []
    ground_truths = []
    questions = []
    correct = 0

    print("\nRunning inference on 100 samples...")
    with torch.no_grad():
        for idx in indices:
            sample = dataset[idx]

            batch = {
                "image_tensor": sample["image_tensor"]
                    .unsqueeze(0).to(device),
                "question_ids": sample["question_ids"]
                    .unsqueeze(0).to(device),
                "attention_mask": sample["attention_mask"]
                    .unsqueeze(0).to(device),
            }

            output = model(batch)

            # Get predicted answer
            pred_idx = output["answer_logits"] \
                .argmax(dim=-1).item()
            pred_answer = idx2ans[pred_idx] if pred_idx < len(idx2ans) else "unknown"
            predictions.append(pred_answer)

            # Get ground truth (most common annotator answer)
            gt_answers = sample["raw_answers"]
            gt_counter = Counter(gt_answers)
            gt_answer = gt_counter.most_common(1)[0][0]
            ground_truths.append(gt_answer)
            questions.append(sample["raw_question"])

            # VQA soft score for this prediction
            gt_count = gt_counter.get(pred_answer, 0)
            score = min(gt_count / 3, 1.0)
            correct += score

    # Results
    vqa_acc = correct / len(indices)
    print(f"\nVQA Soft Accuracy on 100 samples: "
          f"{vqa_acc:.4f} ({vqa_acc*100:.1f}%)")

    # Answer distribution analysis
    pred_counter = Counter(predictions)
    print(f"\nTop 20 PREDICTED answers:")
    print(f"{'Answer':<20} {'Count':>6} {'%':>6}")
    print("-" * 35)
    for ans, count in pred_counter.most_common(20):
        pct = count / len(predictions) * 100
        print(f"{ans:<20} {count:>6} {pct:>5.1f}%")

    gt_counter_all = Counter(ground_truths)
    print(f"\nTop 20 GROUND TRUTH answers:")
    print(f"{'Answer':<20} {'Count':>6} {'%':>6}")
    print("-" * 35)
    for ans, count in gt_counter_all.most_common(20):
        pct = count / len(ground_truths) * 100
        print(f"{ans:<20} {count:>6} {pct:>5.1f}%")

    # Collapse detection
    top1_pred = pred_counter.most_common(1)[0]
    top1_pct = top1_pred[1] / len(predictions) * 100
    print(f"\nAnswer collapse check:")
    print(f"Most predicted answer: '{top1_pred[0]}' "
          f"({top1_pct:.1f}% of predictions)")
    if top1_pct > 20:
        print("⚠️  ANSWER COLLAPSE DETECTED — "
              f"model predicts '{top1_pred[0]}' "
              f"for {top1_pct:.1f}% of questions")
        print("   This is expected with limited training data")
        print("   Document this in your reflection")
    else:
        print("✅ No severe answer collapse detected")

    # Show 10 example predictions
    print(f"\n{'='*70}")
    print("10 EXAMPLE PREDICTIONS")
    print(f"{'='*70}")
    for i in range(10):
        q = questions[i]
        pred = predictions[i]
        gt = ground_truths[i]
        match = "✅" if pred == gt else "❌"
        print(f"{match} Q: {q}")
        print(f"   Predicted: {pred}")
        print(f"   GT:        {gt}")
        print()

    # Save full results
    output_path = ROOT_DIR / "outputs/prediction_inspection.json"
    results = {
        "vqa_accuracy_100_samples": vqa_acc,
        "top_20_predictions": pred_counter.most_common(20),
        "top_20_ground_truths": gt_counter_all.most_common(20),
        "answer_collapse_detected": top1_pct > 20,
        "most_predicted_answer": top1_pred[0],
        "most_predicted_pct": top1_pct,
        "examples": [
            {
                "question": questions[i],
                "predicted": predictions[i],
                "ground_truth": ground_truths[i]
            }
            for i in range(20)
        ]
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Full results saved to {output_path}")

if __name__ == "__main__":
    main()