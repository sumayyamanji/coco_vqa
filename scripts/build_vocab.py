"""Build the answer vocabulary and (optionally) the FAISS retrieval index."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import ROOT_DIR
from src.data.answer_vocab import AnswerVocab


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build answer vocab and/or FAISS index from local VQA v2 annotations"
    )
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument(
        "--min-freq",
        type=int,
        default=9,
        help="Minimum annotation frequency to include an answer (default: 9)",
    )
    p.add_argument("--output", default=None, help="Override vocab output path from config")

    # Index-building arguments
    p.add_argument(
        "--build-index",
        action="store_true",
        help="Also build the FAISS retrieval index from the training split",
    )
    p.add_argument(
        "--index-output",
        default=None,
        help="Override FAISS index output directory from config "
             "(config key: retrieval.faiss_index_path)",
    )
    p.add_argument(
        "--index-batch-size",
        type=int,
        default=64,
        help="Images per embedding batch when building the index (default: 64)",
    )
    p.add_argument(
        "--embedder-device",
        default=None,
        help="Device for CLIPEmbedder, e.g. 'cuda' or 'cpu' (auto-detects by default)",
    )
    return p.parse_args()


def build_vocab(cfg: dict, args: argparse.Namespace) -> None:
    annotations_path = ROOT_DIR / cfg["data"]["annotations_train"]
    out_path = Path(args.output) if args.output else ROOT_DIR / cfg["paths"]["vocab_path"]
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


def build_faiss_index(cfg: dict, args: argparse.Namespace) -> None:
    from src.data.answer_vocab import load_vocab
    from src.data.dataset import VQADataset
    from src.retrieval.embedder import CLIPEmbedder
    from src.retrieval.index import VQAIndex

    index_dir = (
        Path(args.index_output)
        if args.index_output
        else ROOT_DIR / cfg.get("retrieval", {}).get("faiss_index_path", "data/faiss_index")
    )

    vocab_path = Path(args.output) if args.output else ROOT_DIR / cfg["paths"]["vocab_path"]
    if not vocab_path.exists():
        raise FileNotFoundError(
            f"Vocab file not found at {vocab_path}. Run without --build-index first "
            "to create the vocab, or pass --output pointing to an existing file."
        )

    print(f"Loading vocab from {vocab_path} …")
    ans2idx, _idx2ans = load_vocab(vocab_path)

    print("Loading training dataset …")
    # Use minimal transform — we only need raw PIL images for embedding
    train_dataset = VQADataset(
        split="train",
        config=cfg,
        vocab=ans2idx,
        transform=None,
        mode="image_only",
    )
    print(f"  {len(train_dataset):,} training samples")

    vision_backbone: str = cfg.get("model", {}).get(
        "vision_backbone", "openai/clip-vit-large-patch14"
    )
    print(f"Loading CLIPEmbedder ({vision_backbone}) …")
    embedder = CLIPEmbedder(
        model_name=vision_backbone,
        device=args.embedder_device,
        batch_size=args.index_batch_size,
    )

    print(f"Building FAISS index → {index_dir} …")
    VQAIndex.build_from_dataset(
        dataset=train_dataset,
        embedder=embedder,
        save_path=index_dir,
        batch_size=args.index_batch_size,
    )
    print(f"Done. Index saved to {index_dir}/")


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    build_vocab(cfg, args)

    if args.build_index:
        build_faiss_index(cfg, args)


if __name__ == "__main__":
    main()
