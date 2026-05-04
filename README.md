# COCO-VQA


A Visual Question Answering system built on VQAv2 using CLIP + BERT + cross-attention fusion. This is a program for an accessible technology - namely to be deployed for blind and visually impaired users, to ask natural language questions abotu their immediate physical environment without relying on a sighted human assistant. 

A proof-of-concept is built where a user can photograph any object or scene, ask a free-text question, and receive a ranked set of candidate answers with associated confidence scores. 

Visual Question Answering (VQA) is the task of answering a natural-language question about an image. A model receives a raw photograph and an open-ended question such as "How many dogs are in the park?" and must produce a natural-language answer — "two" — from a large candidate vocabulary. 

VQAv2 contains approximately 443,000 questions over 214,000 MS-COCO images, with each question annotated by 10 independent human annotators.

Questions are partitioned into three answer types: yes/no (binary, ~38% of questions), number (integer counting, ~12%), and other (open-ended, ~50%).


## Architecture

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║             VQA MODEL — END-TO-END FORWARD PASS                                 ║
╚══════════════════════════════════════════════════════════════════════════════════╝

  RAW IMAGE (PIL, H×W×3)              RAW QUESTION (string)
         │                                     │
  Resize + centre-crop                  BertTokenizer (WordPiece)
  Normalize to 224×224                  max_length=30, uncased
         │                                     │
         ▼                                     ▼
  ┌─────────────────────────────┐   [CLS] tok₁ tok₂ … [SEP] [PAD]…
  │  IMAGE PATCHING             │   input_ids   (B, 30)
  │                             │   attn_mask   (B, 30)
  │  224×224 px image                          │
  │  ÷ 14px patch size                         │
  │  = 16×16 = 256 patches      │              │
  │                             │              │
  │  ┌──┬──┬──┬──┬──┬──┐        │              ▼
  │  │  │  │  │  │  │  │        │   ┌──────────────────────────┐
  │  ├──┼──┼──┼──┼──┼──┤ 16 rows│   │  TEXT ENCODER (BERT)     │
  │  │  │  │  │  │  │  │        │   │  bert-base-uncased        │
  │  ├──┼──┼──┼──┼──┼──┤        │   │  12 layers, 12 heads     │
  │  │  │  │  │  │  │  │        │   │  hidden_dim = 768         │
  │  └──┴──┴──┴──┴──┴──┘        │   │                          │
  │     16 columns              │   │  Bidirectional self-attn  │
  │  Each patch: 14×14×3 pixels │   │  over all 30 tokens       │
  └─────────────────────────────┘   │                          │
         │                          │  token_emb: (B, 30, 768) │
         ▼                          │  cls_emb:   (B, 768)     │
  ┌─────────────────────────────┐   └──────────────────────────┘
  │  VISION ENCODER (CLIP ViT)  │              │
  │  openai/clip-vit-large-     │              │  tok_emb (B, 30, 768)
  │  patch14                    │              │
  │                             │              │
  │  Each patch → Linear proj   │              │
  │  → 1024-dim embedding       │              │
  │  + learned positional emb   │              │
  │  [CLS] prepended → 257 toks │              │
  │                             │              │
  │  24 transformer layers      │              │
  │  (full self-attention)      │              │
  │                             │              │
  │  Output: (B, 257, 1024)     │              │
  │  Project 1024 → 768 (Linear)│              │
  │  + LayerNorm                │              │
  │                             │              │
  │  patch_emb: (B, 256, 768)   │              │
  │  cls_emb:   (B, 768)        │              │
  └─────────────────────────────┘              │
         │ patch_emb                           │
         ▼                                     │
  ┌──────────────────────────────────┐         │
  │  SCENE GRAPH (optional,          │         │
  │  disabled by default)            │         │
  │                                  │         │
  │  Spatial relations:              │         │
  │    left / right / above /        │         │
  │    below / inside                │         │
  │  Semantic relations:             │         │
  │    holding / wearing / near      │         │
  │                                  │         │
  │  2× attention message-passing    │         │
  │  + spatial positional bias table │         │
  │  + COCO-80 label embeddings      │         │
  │                                  │         │
  │  Output: enriched patch_emb      │         │
  │          (B, 256, 768) residual  │         │
  └──────────────────────────────────┘         │
         │ patch_emb (B, 256, 768)             │
         │                                     │ tok_emb (B, 30, 768)
         └─────────────────┬───────────────────┘
                           ▼
  ╔══════════════════════════════════════════════════════════╗
  ║          CROSS-MODAL FUSION  ×4 layers                   ║
  ║                                                          ║
  ║  ┌────────────────────────────────────────────────────┐  ║
  ║  │  CrossModalBlock (repeated 4×)                     │  ║
  ║  │                                                    │  ║
  ║  │  Q→V  (question tokens attend image patches)       │  ║
  ║  │  query: tok_emb (B, 30, 768)                       │  ║
  ║  │  key / value: patch_emb (B, 256, 768)              │  ║
  ║  │  MHA (8 heads, 96 dims/head)                       │  ║
  ║  │  attn_weights (B, 30, 256) ← saved for heatmaps    │  ║
  ║  │  residual + LayerNorm + FFN (768→3072→768)         │  ║
  ║  │                                                    │  ║
  ║  │  V→Q  (image patches attend question tokens)       │  ║
  ║  │  query: patch_emb (B, 256, 768)                    │  ║
  ║  │  key / value: tok_emb (B, 30, 768)                 │  ║
  ║  │  MHA (8 heads) + pad mask on [PAD] tokens          │  ║
  ║  │  residual + LayerNorm + FFN (768→3072→768)         │  ║
  ║  └────────────────────────────────────────────────────┘  ║
  ║                                                          ║
  ║  After 4 blocks:                                         ║
  ║  text_pooled = mean(tok_emb,   dim=1)  → (B, 768)        ║
  ║  img_pooled  = mean(patch_emb, dim=1)  → (B, 768)        ║
  ║  cat([text_pooled, img_pooled])        → (B, 1536)       ║
  ║  Linear(1536→768) + LayerNorm          → (B, 768)        ║
  ╚══════════════════════════════════════════════════════════╝
                           │
                    fused  (B, 768)
                           │
          ┌────────────────┼──────────────────┐
          ▼                ▼                  ▼
  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐
  │AnswerType    │  │  YesNoHead   │  │  NumberHead         │
  │Classifier    │  │              │  │                     │
  │LayerNorm(768)│  │ Linear(768→2)│  │ Linear(768→50)      │
  │Linear(768→  │  │              │  │                     │
  │  384)        │  │ → [no, yes]  │  │ → integers 0–49     │
  │GELU          │  │  (B, 2)      │  │  (B, 50)            │
  │Dropout(0.1)  │  └──────────────┘  └────────────────────┘
  │Linear(384→3) │          │                  │
  │              │          │ auxiliary        │ auxiliary
  │ → (B, 3)     │          │ heads            │ heads
  │  0: yes/no   │
  │  1: number   │
  │  2: other    │
  │              │
  │ routes       │
  │ predict()    │
  └──────────────┘
          │
          │          ┌──────────────────────────────────┐
          │          │  OpenEndedHead  ← PRIMARY         │
          │          │                                   │
          └─────────►│  LayerNorm(768)                   │
          (all        │  Linear(768→768) → GELU           │
          routing)    │  Dropout(0.1)                    │
                      │  Linear(768→15256)                │
                      │                                   │
                      │  → logits (B, 15256)              │
                      │  softmax → topk(3)                │
                      │  confidence = max(softmax)        │
                      └──────────────────────────────────┘
                                      │
                      ┌───────────────────────────────────┐
                      │  GenerativeHead  (optional)       │
                      │                                   │
                      │  fused → Linear → (B, 1, 768)     │
                      │  (single-token memory)            │
                      │                                   │
                      │  4-layer TransformerDecoder       │
                      │  8 heads, pre-norm, max_len=10    │
                      │                                   │
                      │  Decoding:                        │
                      │  [BOS] → tok₁ → tok₂ → [EOS]     │
                      │  greedy  or  beam search (k=3)    │
                      │                                   │
                      │  → answer token sequence          │
                      └───────────────────────────────────┘
```


1. **VisionEncoder**: OpenAI CLIP ViT-L/14 produces grid patch tokens (frozen during training)
2. **TextEncoder**: BERT-base-uncased encodes the question (frozen during training)
3. **CrossModalFusion**: bidirectional cross-attention over 4 layers; question tokens attend to image patches
4. **AnswerTypeClassifier**: 3-way head predicting yes/no / number / other
5. **YesNoHead**: auxiliary binary head (2 classes)
6. **NumberHead**: auxiliary head for counting questions (0–49)
7. **OpenEndedHead**: primary classification head over 15,256 answer classes; used for training loss
8. **GenerativeHead** *(optional)*: autoregressive decoder for free-form answer generation
9. **SceneGraphGenerator** *(optional, disabled by default)* : GCN to enrich patch tokens with object relations


## Running `reproduce.ipynb` on subset of val images (demo)


- Click 'run all' on `reproduce.ipynb'
- It calls the `best_model.pt` output of the Vision Transformer
- Note in Cell 6, you need to manually add the name of the image under `demo_images`


## Run full pipeline


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

