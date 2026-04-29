# COCO-VQA

A Visual Question Answering system built on VQAv2 using CLIP + BERT + cross-attention fusion.

## Architecture

```
Image ──► CLIP ViT ──────────────────────┐
                      patch tokens        │
                                          ▼
Question ──► BERT ──► token embeds ──► Cross-Attention Fusion ──► MLP ──► answer logits
```

1. **VisionEncoder** — OpenAI CLIP ViT-L/14 produces grid patch tokens
2. **TextEncoder** — BERT-base-uncased encodes the question
3. **SceneGraphEncoder** *(optional)* — GCN enriches patch tokens with object relations
4. **CrossAttentionFusion** — question tokens attend to image patches over 4 layers
5. **AnswerClassifier** — gated MLP projects fused CLS to 3,129 answer logits

## Quick Start

```bash
# 1. Create environment
bash setup.sh
source coco_vqa_env/bin/activate

# 2. Build the answer vocabulary (one-time)
python scripts/build_vocab.py

# 3. A) Full train
python scripts/train.py --config configs/config.yaml


# For testing 1% of the vocab on CPU (before training fully on GPU):

python scripts/train.py --config configs/config.yaml --mode multimodal --debug --no-wandb


# 4. Evaluate
python scripts/evaluate.py --checkpoint checkpoints/checkpoint_epoch0020.pt

# 5. Run the Gradio demo
python demo/app.py
```

## Project Structure

```
coco_vqa/
├── notebooks/          # Exploratory and walkthrough notebooks
├── src/
│   ├── data/           # Dataset, augmentations, answer vocab
│   ├── models/         # VisionEncoder, TextEncoder, Fusion, VQAModel
│   ├── training/       # Trainer, VQALoss, LR scheduler
│   ├── evaluation/     # VQA metric, additional metrics, visualisation
│   └── utils/          # Checkpointing, W&B logger, GradCAM
├── demo/               # Gradio web demo
├── configs/            # config.yaml — all hyperparameters
├── scripts/            # train.py, evaluate.py, build_vocab.py
└── tests/              # pytest unit tests
```

## Configuration

All hyperparameters live in [`configs/config.yaml`](configs/config.yaml).
Key settings:

| Parameter | Value |
|---|---|
| Vision backbone | `openai/clip-vit-large-patch14` |
| Text encoder | `bert-base-uncased` |
| Hidden dim | 768 |
| Fusion layers | 4 |
| Answer classes | 3,129 |
| Batch size | 32 |
| Epochs | 20 |
| Learning rate | 1e-4 |
| FP16 | ✓ |

## Docker

```bash
docker build -t coco-vqa .
docker run -p 7860:7860 coco-vqa
```

## Tests

```bash
pytest tests/ -v
```

## Notebooks

| Notebook | Purpose |
|---|---|
| `01.ipynb` | Initial pipeline test |
| `02_data_exploration.ipynb` | Dataset statistics and visualisation |
| `03_model_walkthrough.ipynb` | Architecture deep-dive with shape checks |
| `04_evaluation_analysis.ipynb` | Accuracy analysis, GradCAM saliency |
