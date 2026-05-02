"""Prepare demo images for reproduce.ipynb.

Run this once before zipping the project:
    python scripts/prepare_demo.py

Selects 25 diverse val images (10 yes/no + 5 number + 10 other), copies them to
demo_images/, and writes demo_images/metadata.json with Q&A pairs for each.
Images are chosen by answer consensus so ground-truth is unambiguous.
"""
from __future__ import annotations

import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ANN_PATH = ROOT / "data/raw/annotations/v2_mscoco_val2014_annotations.json"
Q_PATH   = ROOT / "data/raw/questions/v2_OpenEnded_mscoco_val2014_questions.json"
IMG_DIR  = ROOT / "data/raw/images/val2014"
OUT_DIR  = ROOT / "demo_images"

N_YES_NO = 10
N_NUMBER = 5
N_OTHER  = 10


def _consensus(answers: list[dict]) -> float:
    counts = Counter(a["answer"] for a in answers)
    return counts.most_common(1)[0][1] / len(answers)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    print("Loading val annotations...")
    anns = json.loads(ANN_PATH.read_text(encoding="utf-8"))["annotations"]
    qs   = json.loads(Q_PATH.read_text(encoding="utf-8"))["questions"]
    qid2q = {q["question_id"]: q["question"] for q in qs}

    # Group by answer type, keep only samples whose image exists on disk
    by_type: dict[str, list] = defaultdict(list)
    for ann in anns:
        img_path = IMG_DIR / f"COCO_val2014_{ann['image_id']:012d}.jpg"
        if not img_path.exists():
            continue
        by_type[ann["answer_type"]].append((_consensus(ann["answers"]), ann))

    # Sort by consensus desc so we pick the clearest ground-truth examples
    rng = random.Random(42)
    selected = []
    for at, n in [("yes/no", N_YES_NO), ("number", N_NUMBER), ("other", N_OTHER)]:
        pool = sorted(by_type[at], key=lambda x: x[0], reverse=True)[:500]
        chosen = rng.sample(pool, min(n, len(pool)))
        selected.extend(entry[1] for entry in chosen)

    metadata = []
    for ann in selected:
        img_id  = ann["image_id"]
        qid     = ann["question_id"]
        answers = [a["answer"] for a in ann["answers"]]
        src = IMG_DIR / f"COCO_val2014_{img_id:012d}.jpg"
        dst = OUT_DIR / f"COCO_val2014_{img_id:012d}.jpg"
        shutil.copy2(src, dst)
        metadata.append({
            "image_id":           img_id,
            "question_id":        qid,
            "filename":           dst.name,
            "question":           qid2q[qid],
            "answers":            answers,
            "most_common_answer": Counter(answers).most_common(1)[0][0],
            "answer_type":        ann["answer_type"],
            "consensus":          round(_consensus(ann["answers"]), 3),
        })

    (OUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    by_t = Counter(m["answer_type"] for m in metadata)
    avg_con = sum(m["consensus"] for m in metadata) / len(metadata)
    print(f"Done — {len(metadata)} images written to {OUT_DIR}/")
    print(f"  yes/no: {by_t['yes/no']}  number: {by_t['number']}  other: {by_t['other']}")
    print(f"  Average answer consensus: {avg_con:.2f}")
    print(f"\nNext step: zip the project, then open reproduce.ipynb")


if __name__ == "__main__":
    main()
