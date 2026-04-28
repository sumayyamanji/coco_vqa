"""Build the answer vocabulary from local VQA v2 training annotations."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.data.answer_vocab import AnswerVocab


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build answer vocab from local VQA v2 annotations")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--min-freq", type=int, default=9,
                   help="Minimum annotation frequency to include an answer (default: 9)")
    p.add_argument("--output", default=None, help="Override output path from config")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    annotations_path = Path(cfg["data"]["annotations_train"])
    out_path = Path(args.output or cfg["paths"]["vocab_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not annotations_path.exists():
        raise FileNotFoundError(
            f"Annotations not found at {annotations_path}.\n"
            "Unzip v2_Annotations_Train_mscoco.zip into data/raw/annotations/ first."
        )

    print(f"Loading annotations from {annotations_path} …")
    vocab = AnswerVocab.build_from_annotations(annotations_path, min_freq=args.min_freq)
    vocab.save(out_path)
    print(f"Saved {len(vocab):,} answers to {out_path}")


if __name__ == "__main__":
    main()
