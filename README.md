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

## Running on Google Colab

Recommended for training: use an **A100 runtime** (Colab Pro) for ~7–10 hr full training, or a free T4 for ~35–40 hr.

**1. Mount Google Drive and upload your data**

Upload the entire `coco_vqa/` folder to your Drive, keeping the `data/raw/` structure intact.

```python
from google.colab import drive
drive.mount("/content/drive")
%cd /content/drive/MyDrive/coco_vqa
```

**2. Install dependencies**

```python
!pip install torch torchvision transformers tqdm pyyaml Pillow matplotlib
```

**3. Switch to A100 runtime**

Runtime → Change runtime type → A100 GPU

**4. Run training**

```python
# Smoke-test (1% data, ~4 min on A100)
!python scripts/train.py --config configs/config.yaml --mode multimodal --debug --no-wandb

# Full training (~7–10 hr on A100)
!python scripts/train.py --config configs/config.yaml --mode multimodal --no-wandb
```

**5. Evaluate**

```python
!python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --no-wandb
```

> **Tips:**
> - `answer_vocab.json` can be built locally first and uploaded — saves ~5 min.
> - Enable W&B by removing `--no-wandb` and entering your API key when prompted.
> - Colab sessions time out; use `--resume checkpoints/checkpoint_epochXX.pt` to continue.

---

## Running on Azure

Recommended instance: **Standard_NC24ads_A100_v4** (1× A100 40 GB, ~$3.50/hr).
A **Standard_NC6s_v3** (V100 16 GB) is cheaper (~$0.90/hr) and suits fp16 training.

**1. Provision a VM**

```bash
az vm create \
  --resource-group my-rg \
  --name coco-vqa-vm \
  --image microsoft-dsvm:ubuntu-2004:2004:latest \
  --size Standard_NC24ads_A100_v4 \
  --admin-username azureuser \
  --generate-ssh-keys
```

**2. SSH in and upload the project**

```bash
ssh azureuser@<vm-public-ip>

# From your local machine — upload project and data
rsync -avz coco_vqa/ azureuser@<vm-public-ip>:~/coco_vqa/
```

**3. Install dependencies on the VM**

```bash
cd ~/coco_vqa
pip install torch torchvision transformers tqdm pyyaml Pillow matplotlib
```

**4. Run training in a persistent session**

```bash
# Use tmux or screen so the job survives SSH disconnects
tmux new -s train

python scripts/train.py --config configs/config.yaml --mode multimodal --no-wandb

# Detach: Ctrl+B then D  |  Reattach: tmux attach -t train
```

**5. Copy results back**

```bash
# From your local machine
rsync -avz azureuser@<vm-public-ip>:~/coco_vqa/checkpoints/ ./checkpoints/
```

> **Tips:**
> - The Azure DSVM image has CUDA and conda pre-installed.
> - Stop the VM when not in use (`az vm deallocate`) to avoid charges.
> - `configs/config.yaml` already has `fp16: true` — no changes needed for GPU runs.

---

## Expected Training Times

| Hardware | Full 20 epochs | Debug (1% data) |
|---|---|---|
| CPU | ~weeks | ~2–3 hr |
| Colab T4 (free) | ~35–40 hr | ~20 min |
| Colab A100 (Pro) | ~7–10 hr | ~4 min |
| Azure V100 16 GB | ~15–20 hr | ~8 min |
| Azure A100 40 GB | ~7–10 hr | ~4 min |

---

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
