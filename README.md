# COCO-VQA

A Visual Question Answering system built on VQAv2 using CLIP + BERT + cross-attention fusion.

## Architecture

```
Image ──► CLIP ViT-L/14 ──► patch tokens ──────────────────────┐
                                                                 ▼
Question ──► BERT-base ──► token embeddings ──► CrossModalFusion (4 layers)
                                                                 │
                                                    ┌────────────▼────────────┐
                                                    │      fused CLS token    │
                                                    └──┬──────┬───────┬───────┘
                                                       │      │       │
                                               AnswerType  YesNo  Number  OpenEnded
                                               (3 classes) (2)    (0–49)  (15,256)
```

1. **VisionEncoder** — OpenAI CLIP ViT-L/14 produces grid patch tokens (frozen during training)
2. **TextEncoder** — BERT-base-uncased encodes the question (frozen during training)
3. **CrossModalFusion** — bidirectional cross-attention over 4 layers; question tokens attend to image patches
4. **AnswerTypeClassifier** — 3-way head predicting yes/no / number / other
5. **YesNoHead** — auxiliary binary head (2 classes)
6. **NumberHead** — auxiliary head for counting questions (0–49)
7. **OpenEndedHead** — primary classification head over 15,256 answer classes; used for training loss
8. **GenerativeHead** *(optional)* — autoregressive decoder for free-form answer generation
9. **SceneGraphGenerator** *(optional, disabled by default)* — GCN to enrich patch tokens with object relations

## Quick Start

```bash
# Git Bash / bash
source coco_vqa_env/Scripts/activate

# Windows Command Prompt
coco_vqa_env\Scripts\activate.bat

# PowerShell
coco_vqa_env\Scripts\Activate.ps1


# 2. Build the answer vocabulary (one-time)
python scripts/build_vocab.py


# 3. Check if PyTorch can see your GPU 
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# 3a. Run 10% of vocab - what I did
python scripts/train.py --config configs/config.yaml --mode multimodal --debug

# 3b. Full train (did not do this myself for this task)
python scripts/train.py --config configs/config.yaml

# NB: To stop Windows sleeping during the run, put this into Cmd: 
powercfg /change standby-timeout-ac 0
powercfg /change monitor-timeout-ac 0

# NB: If at any point, device gets disconnected, just run the above again. It'll automatically find the latest checkpoint and resume from the latest 

# 4. Evaluate 
# 4a. On everything
python scripts/evaluate.py --checkpoint outputs/checkpoints/best_model.pt

# 4b. On 3000 samples (what I did)
python scripts/evaluate.py --checkpoint outputs/checkpoints/best_model.pt --max-samples 3000


# 5. Run the Gradio demo
python demo/app.py
```

## Run Baselines (BERT model ie text-only)
```bash
python baselines/train_baselines.py --config baselines/configs/baselines_config.yaml
```

Run baselines evaluation 
```bash
python baselines/evaluate_baselines.py
```

**NB: By default, this also runs on the same 10% of the training data as the main model. Evaluation uses 3,000 stratified validation samples (seed=42, maintaining ~38% yes/no / 12% number / 50% other distribution) — same sample for both models for a fair direct comparison.**

## Project Structure

```
coco_vqa/
├── notebooks/          # Exploratory and walkthrough notebooks
├── src/
│   ├── data/           # Dataset, augmentations, answer vocab
│   ├── models/         # VisionEncoder, TextEncoder, Fusion, VQAModel
│   ├── training/       # Trainer, VQALoss, LR scheduler
│   ├── evaluation/     # VQA metric, additional metrics, visualisation
│   ├── retrieval/      # FAISS-based image retrieval (embedder, index, retriever)
│   └── utils/          # Checkpointing, W&B logger, GradCAM
├── demo/               # Gradio web demo: assets/, examples/, app.py
├── demo_images/        # 25 COCO val2014 images + metadata.json (used by prepare_demo.py)
├── configs/            # config.yaml — all hyperparameters
└── scripts/            # train.py, evaluate.py, build_vocab.py, check_output.py, demo_inference.py, prepare_demo.py

baselines/outputs/
├── checkpoints/
│   └── text_only_bert/
│       ├── checkpoint_epoch01_acc0.xxxx.pt   
│       ├── checkpoint_epoch02_acc0.xxxx.pt
│       ├── checkpoint_epoch03_acc0.xxxx.pt
│       ├── best_model.pt                      
│       └── latest.pt                         
│
├── plots/                                    
│   ├── comparison_bar.png
│   ├── language_bias.png
│   ├── training_curve.png
│   └── question_lengths.png                  
│
├── results.json                              
│                                               
├── master_results.json                       
│                                               
└── training_log.json                         
```

## Configuration

All hyperparameters live in [`configs/config.yaml`](configs/config.yaml).

## Docker

```bash
docker build -t coco-vqa .
docker run -p 7860:7860 coco-vqa
```

## How I ran the code

- Ran it on --debug mode ie only on 10% of data. Therefore optimizing for runtime, without reducing the model architecture (the full CLIP ViT-Large model)

- Didn't decide to reduce the model size, as this would mean rewriting model section and getting worse results for no good reason

- 10% of real data still gives you meaningful accuracy numbers to report