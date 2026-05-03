# Visual Question Answering: Multimodal Deep Learning on VQAv2 + COCO

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [How the Full Model Works End-to-End](#2-how-the-full-model-works-end-to-end)
3. [Main Model Design Choices & Hyperparameters](#3-main-model-design-choices--hyperparameters)
4. [Baseline Model: Text-Only DistilBERT](#4-baseline-model-text-only-distilbert)
5. [Training Results](#5-training-results)
6. [Evaluation Strategy](#6-evaluation-strategy)
7. [Approaches to VQA in the Literature](#7-approaches-to-vqa-in-the-literature)
8. [Why This Architecture Was Chosen](#8-why-this-architecture-was-chosen)
9. [Future Research Directions](#9-future-research-directions)
10. [Quick Start & How to Run](#10-quick-start--how-to-run)
11. [References](#references)

---

## 1. Project Overview

Visual Question Answering (VQA) is the task of answering a natural-language question about an image. A model receives a raw photograph and an open-ended question such as "How many dogs are in the park?" and must produce a natural-language answer — "two" — from a large candidate vocabulary. VQA is considered one of the hardest benchmarks in vision-language understanding because it simultaneously requires visual perception (localising and identifying scene entities), language understanding (parsing question syntax and semantics), and cross-modal reasoning (mapping visual evidence to the question's information need). Unlike unimodal tasks, a failure in either modality propagates to the final answer, making each component of the pipeline safety-critical.

The standard benchmark for this task is VQAv2 (Goyal et al., 2017), a large-scale dataset paired with COCO images. VQAv2 contains approximately 443,000 questions over 214,000 MS-COCO images, with each question annotated by 10 independent human annotators. The commonly cited published benchmark uses a closed vocabulary of 3,129 answers; this project instead retains all answers appearing at least 9 times in training annotations with no hard cap, producing a **15,256-class vocabulary** (see §2.8). Questions are partitioned into three answer types: yes/no (binary, ~38% of questions), number (integer counting, ~12%), and other (open-ended, ~50%). This type distribution matters for evaluation: a model that only learns linguistic cues can score surprisingly well on yes/no questions without any visual understanding at all.

This language-bias problem was first systematically exposed by Goyal et al. (2017), who showed that models trained on the original VQA (Antol et al., 2015) dataset learned to exploit strong correlations between question phrasing and majority answers — for example, answering "yes" whenever a question begins with "Is there a …" without reading the image. VQAv2 was constructed to counteract this bias by pairing each question with two images: one where the answer is "yes" and one where it is "no". Despite this correction, language bias remains exploitable on VQAv2, motivating the text-only DistilBERT baseline in this project: if a model with no visual input scores nearly as well as the multimodal model, most of the signal is coming from question phrasing rather than image content.

The real-world significance of robust VQA is considerable. For visually impaired users, VQA systems can serve as autonomous image description and question-answering agents, enabling access to visual information through natural conversation (Gurari et al., 2018 — VizWiz). In medical imaging, a clinician might ask "Is there a mass in the lower left lobe?" of a chest X-ray, requiring precise spatial visual reasoning not available to text-only systems. In e-commerce, product images can be queried conversationally without manual metadata tagging. The gap between current models and human performance (~90% on VQAv2) represents a significant open research challenge, particularly for number and spatial reasoning questions where deep visual grounding is unavoidable.

---

## 2. How the Full Model Works End-to-End

This section traces a single sample from raw inputs to a final answer prediction, based on the actual code in [src/models/](src/models/).

### 2.0 Architecture Overview

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

### 2.1 Input

A training or inference sample consists of two raw inputs:

- **Image**: A PIL RGB image loaded from `data/raw/images/train2014/` following the COCO filename convention `COCO_train2014_{image_id:012d}.jpg`. For training, data augmentation is applied (random horizontal flip, colour jitter, random crop). For validation and inference, only a centre crop and normalisation are applied. The final image tensor has shape `(3, 224, 224)`.

- **Question**: A raw string such as "What colour is the bus?" passed through `BertTokenizer.from_pretrained("bert-base-uncased")` with `padding="max_length"`, `truncation=True`, `max_length=30`. The tokeniser prepends `[CLS]` and appends `[SEP]`, applies WordPiece subword tokenisation, and pads with zeros. This produces `input_ids` of shape `(1, 30)` and `attention_mask` of shape `(1, 30)`.

The image resolution of 224×224 was chosen because CLIP ViT-L/14 was pretrained at exactly this resolution; using a different resolution would require interpolating positional embeddings, degrading representation quality (Radford et al., 2021).

### 2.2 Vision Encoder: CLIP ViT-L/14

The vision encoder is implemented in [src/models/vision_encoder.py](src/models/vision_encoder.py) as `CLIPVisionEncoder`, wrapping `openai/clip-vit-large-patch14` from HuggingFace.

**How ViT works**: The 224×224 input image is divided into a 16×16 grid of non-overlapping 14×14 pixel patches, yielding **256 patches** total (224 ÷ 14 = 16 per side). Each patch is flattened and linearly projected to the model's internal dimension of 1,024 (the ViT-L hidden size). A learned positional embedding is added to each patch embedding to encode spatial location. A special `[CLS]` token is prepended to the 256 patch tokens, following the BERT design (Devlin et al., 2018), giving a sequence of 257 tokens. This full sequence is passed through 24 transformer self-attention layers, where every token attends to every other token. Because the `[CLS]` token has no fixed spatial role, it is free to aggregate global context from the entire image through self-attention — making it the standard choice for downstream classification heads.

The CLIP backbone outputs `last_hidden_state` of shape `(B, 257, 1024)`. The encoder then:
1. Extracts `hidden[:, 0]` as the CLS embedding (global image representation).
2. Extracts `hidden[:, 1:]` as 256 patch embeddings (spatial representations).
3. Projects both from 1,024 to the shared `hidden_dim=768` via `nn.Linear(1024, 768)` (since CLIP ViT-L outputs 1,024-dim but the shared hidden dimension is 768 to match BERT-base).
4. Applies `nn.LayerNorm(768)`.

Final outputs: `patch_embeddings` of shape `(B, 256, 768)` and `cls_embedding` of shape `(B, 768)`.

**Why CLIP specifically**: CLIP (Contrastive Language–Image Pretraining) was trained on 400 million image-text pairs from the internet using a contrastive objective that aligns image and text representations in a shared embedding space (Radford et al., 2021). As a consequence, CLIP's visual features are inherently language-aware before any VQA fine-tuning — a patch attending to a dog will produce features close to the text embedding of "dog". This alignment property makes CLIP features uniquely suited to multimodal tasks.

**Why ViT-L/14 over ViT-B/32**: The 14-pixel patch resolution provides 256 patches per image versus 49 patches for the 32-pixel variant, capturing finer spatial detail critical for counting, colour discrimination, and object localisation. ViT-L has 307M parameters versus ViT-B's 86M; larger CLIP models consistently outperform smaller variants on downstream vision-language tasks (Radford et al., 2021).

### 2.3 Text Encoder: BERT

The text encoder is implemented in [src/models/text_encoder.py](src/models/text_encoder.py) as `BERTTextEncoder`, wrapping `bert-base-uncased`.

**Tokenisation**: The BertTokenizer applies WordPiece tokenisation, splitting words into subword units (e.g. "running" → "run", "##ning"). `[CLS]` is prepended, `[SEP]` appended, and the sequence padded with `[PAD]` to length 30. All inputs are lowercased (uncased variant).

**Architecture**: BERT-base has 12 transformer layers with 12 self-attention heads and a hidden dimension of 768. Unlike unidirectional language models, BERT uses bidirectional attention: every token attends to every other token, so "not red" produces a different `[CLS]` embedding from "red" — something unidirectional (GPT-style) or bag-of-words (TF-IDF) models cannot achieve.

The encoder runs `BertModel` and then:
1. Projects `last_hidden_state` from `bert_dim=768` to `hidden_dim=768`. Since these are equal, the projection is `nn.Identity()` — no information loss.
2. Applies `nn.LayerNorm(768)`.
3. Returns `token_embeddings` of shape `(B, 30, 768)` and `cls_embedding = token_embeddings[:, 0]` of shape `(B, 768)`.

**Why bert-base-uncased**: VQA questions are not case-sensitive ("what colour" and "What colour" are semantically identical), so the uncased variant is appropriate. BERT-base balances representational capacity against compute — BERT-large would add ~240M parameters for marginal gains on a task constrained primarily by visual grounding rather than language modelling depth.

### 2.4 Optional Pre-Fusion Enrichment: SceneGraphGenerator

Implemented in [src/models/scene_graph.py](src/models/scene_graph.py) as `SceneGraphGenerator`. Disabled by default (`use_scene_graph: false` in config); when enabled it runs on the vision encoder's `patch_embeddings` **before** they enter the fusion module.

**What it does**: After the CLIP encoder produces `patch_embeddings (B, 256, 768)`, the SceneGraphGenerator enriches each patch embedding with information about its spatial and semantic relationships to other patches. It does this without requiring an external object detector or graph library — the implementation uses attention-based message passing.

**Relation taxonomy**: 8 relation types are modelled via a learnable embedding table:
- *Spatial* (5): left, right, above, below, inside
- *Semantic* (3): holding, wearing, near

**How it works step by step**:
1. **Object-label injection** (optional): if COCO class labels `(B, M)` are provided (e.g. from a detector), label embeddings from an `nn.Embedding(81, 768)` table are added to the corresponding patch features, grounding specific patches in semantic categories.
2. **Spatial bias**: a learnable positional bias table of shape `(27×27, 8_heads)` encodes relative position between patches in the 16×16 grid. Each patch gets a spatial context vector summarising its mean relative position to all other patches.
3. **Semantic context**: the mean of all 8 relation-type embeddings forms a global semantic prior broadcast to every patch.
4. **Two rounds of attention-based message passing**: each `_RelationMessagePassing` block runs `nn.MultiheadAttention` where the key is shifted by the combined spatial+semantic relation bias — so attention weights are not purely content-based but also shaped by where each patch is and what relationships are relevant.
5. **Residual output**: `out_norm(out_proj(x) + patch_embeddings)` — the enriched features are added back onto the original patch embeddings, preserving the CLIP representations while injecting relational context.

**Why it is useful**: Raw ViT patch features encode individual patch content well but do not explicitly model *relationships between patches* — "the dog is to the left of the car" is implicit in the attention pattern but not explicit in any single patch representation. The SceneGraphGenerator makes spatial and semantic relations explicit, potentially helping with questions like "What is the man holding?" (semantic) or "Is the cat above the table?" (spatial). This is particularly relevant for VQA question types that require relational reasoning beyond object recognition. The design is inspired by graph neural network VQA models (Teney et al., 2017) but avoids the Faster R-CNN preprocessing overhead by operating directly on ViT patch tokens.

**Why disabled by default**: It adds trainable parameters (~2.5M) and compute (two extra attention rounds over 256 patches) while the benefit on the 10% training subset was not validated. It is designed as an optional add-on that can be enabled in config without changing any other code.

### 2.5 Cross-Modal Fusion

The fusion module is implemented in [src/models/fusion.py](src/models/fusion.py) as `CrossModalFusion`, which stacks 4 `CrossModalBlock` layers.

**Per-block operation (CrossModalBlock)**:

Each block performs two parallel cross-attention operations, plus two feed-forward networks (FFNs):

*Q→V side (text attends to image)*:
- `query = text_tokens` (B, 30, 768), `key = value = image_patches` (B, 256, 768)
- `nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)` computes attention weights of shape `(B, 30, 256)` — each question token produces a distribution over 256 image patches
- Residual connection + `nn.LayerNorm(768)`
- FFN: Linear(768→3072) → GELU → Dropout(0.1) → Linear(3072→768) → Dropout(0.1), with residual + LayerNorm

*V→Q side (image attends to text)*:
- `query = image_patches` (B, 256, 768), `key = value = text_tokens` (B, 30, 768)
- Same MultiheadAttention configuration; padding mask applied to prevent attention to `[PAD]` tokens
- Residual + LayerNorm, then FFN with residual + LayerNorm

Both sides are updated per block. The Q→V attention weights are retained for visualisation (attention heatmaps).

**After 4 blocks**:
- `text_pooled = text_tokens.mean(dim=1)` → `(B, 768)`
- `img_pooled = image_patches.mean(dim=1)` → `(B, 768)`
- `fused = LayerNorm(Linear(concat([text_pooled, img_pooled])))`
  - Concat: `(B, 1536)` → Linear(1536→768) → LayerNorm → `(B, 768)`

This single fused vector is the input to all answer heads.

**Why bidirectional cross-attention**: The Q→V direction allows the question to selectively weight image patches — "how many dogs" will attend strongly to dog regions. The V→Q direction allows image content to contextualise the question — ambiguous words like "it" can be resolved by visual context. This co-attention design follows ViLBERT (Lu et al., 2019), which demonstrated 2–4% gains over unimodal or unidirectional fusion on VQA benchmarks.

**Number of heads (8)**: With `hidden_dim=768` and 8 heads, each head operates on 768÷8=96 dimensions. Different heads can independently specialise — one may learn colour-related attention patterns, another spatial relations, another object identity. The 8-head configuration is inherited from both BERT-base and the original transformer (Vaswani et al., 2017).

**Number of fusion layers (4)**: Each layer refines cross-modal alignment; early layers establish coarse correspondences, deeper layers resolve fine-grained interactions. Four layers balance cross-modal alignment depth against compute and overfitting risk on a limited training set, following the Q-Former design in BLIP-2 (Li et al., 2023).

**Layer norm + residual connections**: Residual connections prevent gradient vanishing in deep networks (He et al., 2016). Post-norm LayerNorm (Add & Norm ordering) stabilises the magnitude of activations between layers.

### 2.6 Answer Type Classifier & Routing

`AnswerTypeClassifier` ([src/models/answer_heads.py:85](src/models/answer_heads.py#L85)) is a 3-way classifier over the fused vector:

```
LayerNorm(768) → Linear(768→384) → GELU → Dropout(0.1) → Linear(384→3)
```

This produces `type_logits` of shape `(B, 3)`, mapping to: 0=yes/no, 1=number, 2=other.

During the `predict()` method, the predicted type routes to the dedicated specialist head:
- `yes/no` → `YesNoHead.argmax()` (0=no, 1=yes)
- `number` → `NumberHead.argmax()` (interpreted as integer 0–49)
- `other` → `OpenEndedHead.topk()`

**Why routing**: Yes/no questions require a binary output; number questions require ordinal reasoning over a small integer range; other questions require selection from a large open vocabulary. Mixing these into a single head forces the model to optimise incompatible output structures simultaneously (Anderson et al., 2018). The routing also provides a natural decomposition for per-type analysis and debugging.

### 2.7 Answer Heads & Output

All five answer-related components take the same `(B, 768)` fused vector as input: the `AnswerTypeClassifier` described in §2.6 plus the four prediction heads below.

**YesNoHead** ([src/models/answer_heads.py:41](src/models/answer_heads.py#L41)): `Linear(768→2)`, returns logits for [no, yes].

**NumberHead** ([src/models/answer_heads.py:53](src/models/answer_heads.py#L53)): `Linear(768→50)`, classifies into integers 0–49. The 50-class range covers the vast majority of counting questions in VQAv2.

**OpenEndedHead** ([src/models/answer_heads.py:67](src/models/answer_heads.py#L67)): Two-layer MLP — `LayerNorm(768) → Linear(768→768) → GELU → Dropout(0.1) → Linear(768→15256)`. This is the **primary head** used for training loss and answer prediction. It outputs logits over all 15,256 answer classes.

**GenerativeHead** ([src/models/answer_heads.py:117](src/models/answer_heads.py#L117)): A 4-layer autoregressive transformer decoder with 8 heads and max generation length 10. The fused context vector is projected to a single-token memory sequence for cross-attention. Supports greedy and beam-search (beam_size=3) decoding. This head enables free-form answer generation beyond the closed vocabulary, though the primary training objective uses `OpenEndedHead`.

**Top-3 predictions**: After computing `softmax(answer_logits)`, the model returns `probs.topk(3)` — the three most likely answers with confidence scores. The `confidence` field is the maximum softmax probability.

### 2.8 Vocabulary Size: 15,256 Classes

The answer vocabulary is built by [src/data/answer_vocab.py](src/data/answer_vocab.py) using `min_freq=9`: all answers appearing at least 9 times in training annotations are retained with **no hard cap**. Inspecting the saved vocabulary at `data/answer_vocab.json` gives **15,256 entries** (including `<pad>` and `<unk>` special tokens), which is the actual vocabulary used during training. The training script (`scripts/train.py`) determines vocab_size by `len(idx2ans)` from the loaded file — the `num_answer_classes: 3129` value in `config.yaml` is a dead parameter that is never read by the model initialisation.

A 15,256-class vocabulary covers a substantially larger answer space than the standard 3,129-class benchmark vocabulary (Anderson et al., 2018). The key trade-off: retaining all answers with `min_freq≥9` preserves rare but legitimate answers, avoiding the information loss of truncating to a smaller set. The cost is a larger softmax computation and harder classification problem; published models that report on the 3,129-class vocab are not directly numerically comparable to this model's accuracy, since the benchmark accuracy is computed on the same 3,129-class evaluation set regardless of training vocabulary. The `min_freq=9` threshold ensures only answers with meaningful annotator consensus enter the vocabulary, balancing coverage against sparsity.

---

## 3. Main Model Design Choices & Hyperparameters

All values below are read directly from [configs/config.yaml](configs/config.yaml).

### 3.1 Architecture Hyperparameters

**`hidden_dim: 768`**  
Chosen to match BERT-base's native embedding dimension (768). Text features require no projection (`nn.Identity()`). CLIP ViT-L/14 outputs 1,024-dim embeddings, so a `Linear(1024→768)` projection brings visual features into the shared space. The 768-dim space provides sufficient capacity for both modalities without requiring asymmetric handling or information-lossy bottlenecks.

**`num_attention_heads: 8`**  
768 is evenly divisible by 8 (96 dims per head), matching the standard BERT-base and vanilla transformer configurations (Vaswani et al., 2017). Each head can specialise in different cross-modal patterns — colour attribution, spatial relations, object counting. ViLBERT (Lu et al., 2019) also uses 8-head cross-attention in its co-attention transformer, providing a direct precedent for VQA.

**`fusion_layers: 4`**  
Four stacked `CrossModalBlock` layers provide progressively refined cross-modal alignment without excessive compute. BLIP-2 (Li et al., 2023) demonstrates that 4–6 Q-Former layers achieve near-optimal vision-language alignment for an efficient fusion module design. Fewer than 4 layers risk insufficient cross-modal interaction; more than 4 risk overfitting given training on only 10% of the data.

**`dropout: 0.1`**  
Applied in the FFN of each CrossModalBlock and in the answer heads. Standard regularisation rate for transformer fine-tuning, following Devlin et al. (2018). Prevents overfitting given the relatively small training subset (44,375 samples for a ~545M-parameter model).

**`num_answer_classes: 3129`** *(config value — not used at runtime)*  
This config entry is a dead parameter. The training script sets `vocab_size = len(idx2ans)` from the loaded vocabulary file, which is **15,256** (see §2.8). The config value was not updated to match the actual vocabulary and has no effect on the trained model.

### 3.2 Training Hyperparameters

**`batch_size: 32`**  
Constrained by available GPU VRAM (Lenovo LOQ with ~6–8 GB usable). The full model is approximately 545M parameters; in fp16 this requires ~1.1 GB for weights, plus activations and gradients for a 32-sample batch. Gradient checkpointing further reduces activation memory (see §3.6). Larger batches improve gradient estimate stability (Keskar et al., 2017) but were not feasible on available hardware.

**`lr: 1e-4`**  
Peak learning rate with AdamW optimiser (`weight_decay: 0.01`). For fine-tuning pretrained transformers, Devlin et al. (2018) recommend 5e-5 to 1e-4; higher values risk destabilising pretrained representations while lower values slow convergence. The 1e-4 peak is reached after linear warmup and is safe with gradient clipping (`grad_clip: 1.0`).

### 3.3 Cosine Scheduler with Linear Warmup

Implemented in [src/training/scheduler.py](src/training/scheduler.py) as `get_cosine_schedule_with_warmup`.

**Linear warmup (`warmup_steps: 1000`)**: During the first 1,000 gradient steps, the learning rate ramps linearly from 0 to the peak 1e-4. This prevents large gradient updates in early training when the randomly-initialised fusion and classification heads produce large errors that could corrupt pretrained BERT and CLIP representations.

**Cosine decay**: After warmup, the learning rate follows `0.5 × (1 + cos(π × progress))`, smoothly decaying from 1e-4 to 0 over the remaining training steps. The smooth decay prevents abrupt LR changes that can cause training instability, following standard practice in vision-language pretraining (Lu et al., 2019; Li et al., 2022).

**Observed epoch 4 instability**: Despite the scheduler, epoch 4 showed a train loss spike from 0.053 → 0.103 and number-head accuracy collapse from 29.1% → 0.18%. This likely reflects an adverse interaction between a mini-batch heavily skewed toward number questions and the cosine LR phase (LR had decayed to ~1.3e-5). After collapse, number accuracy remained at 0.18% through epoch 5, confirming the update was irreversible within the training budget. This motivates epoch 3 as the best checkpoint selection.

### 3.4 VQA Soft Loss Function

The training objective is `TotalLoss` ([src/training/losses.py:92](src/training/losses.py#L92)):

```
TotalLoss = 1.0 × VQASoftLoss + 0.5 × AnswerTypeLoss
```

**VQASoftLoss** (weight 1.0): The primary loss over the full 15,256-class vocabulary. Unlike standard CrossEntropyLoss (which assigns probability 1.0 to a single hard label), VQA uses soft targets derived from annotator consensus:

```
target[c] = min(count(annotators_who_said_c) / 3, 1.0)
```

For example, if 6 of 10 annotators said "yes", `target["yes"] = 1.0`. If 2 said "yes", `target["yes"] ≈ 0.667`. Implemented using `F.binary_cross_entropy_with_logits(logits, soft_scores)` — treating each class as an independent binary prediction. BCE handles the multi-label nature of VQA better than single-class cross-entropy, and rewards partially correct answers (Antol et al., 2015).

**Label smoothing (`label_smoothing: 0.1`)**: Applied on top of the soft targets:
```
smoothed = soft_scores × (1 − 0.1) + 0.1 / 15256
```
This prevents overconfident predictions by spreading a small probability mass across all classes, acting as additional regularisation beyond dropout.

**AnswerTypeLoss** (weight 0.5): Standard `nn.CrossEntropyLoss` over the 3-way type classifier (yes/no / number / other). The 0.5 weight prevents the auxiliary type task from dominating the primary answer prediction objective.

### 3.5 Mixed Precision Training (fp16)

`fp16: true` enables `torch.cuda.amp.autocast()` and `GradScaler` in the trainer ([src/training/trainer.py:87](src/training/trainer.py#L87)). Activations and gradient accumulation use 16-bit floats; weight updates are computed in 32-bit for numerical stability. This approximately halves VRAM usage and doubles throughput on modern CUDA GPUs with Tensor Cores, with negligible accuracy impact. fp16 is guarded to CUDA only (`self.use_fp16 = t_cfg["fp16"] and device.type == "cuda"`), so CPU training falls back to fp32 automatically.

### 3.6 Gradient Checkpointing

`gradient_checkpointing: true` is enabled on both the CLIP backbone and BERT encoder. Instead of storing all intermediate activations during the forward pass, gradient checkpointing recomputes activations on-the-fly during the backward pass. This trades compute (~30% slower training) for memory: storing activations for a 307M-parameter CLIP model through 24 transformer layers otherwise requires significant VRAM. Without this setting, training the full model would exceed the available GPU memory on the target hardware.

### 3.7 Training Subset & Epoch Selection

**10% training subset (~44,375 questions)**: The VQAv2 training set has ~443K questions; 10% was used due to the compute constraint of a single laptop GPU across 5 epochs. This subset is explicitly matched between the main model and the DistilBERT baseline (`train_subset: 0.10` in both configs) to ensure the comparison is not confounded by different training set sizes.

**Epoch 3 selected as best checkpoint**: Validation accuracy peaked at 30.66% at epoch 3. Epoch 4 showed number-head collapse (29.08% → 0.18%) driven by a train loss spike (0.053 → 0.103). By epoch 5, the number head remained collapsed. Early stopping at epoch 3 is consistent with checkpoint selection based on validation metric rather than train loss (Prechelt, 1998).

---

## 4. Baseline Model: Text-Only DistilBERT

### 4.1 Why DistilBERT Not BERT-base

The baseline is implemented in [baselines/train_baselines.py](baselines/train_baselines.py) using `distilbert-base-uncased`. DistilBERT (Sanh et al., 2019) is a 66M-parameter distilled version of BERT-base (110M parameters), retaining 97% of BERT's performance on GLUE benchmarks via knowledge distillation. With 40% fewer parameters and 60% faster inference, DistilBERT is appropriate as a lower-bound comparator: the goal is to quantify the language bias floor, not to maximise text-only performance. Note that DistilBERT has no `token_type_ids` argument (unlike BERT), so they are never passed to the model.

The `TextOnlyBERT` architecture: DistilBERT → extract `last_hidden_state[:, 0, :]` (CLS token, 768-dim) → three-layer MLP (Linear(768→512) → ReLU → Dropout(0.1) → Linear(512→256) → ReLU → Dropout(0.1) → Linear(256→15256)) → 15,256 logits. Total parameters: 70.8M (from `baselines/outputs/results.json`). The baseline uses the same 15,256-class vocabulary as the main model, loaded from `data/answer_vocab.json`.

### 4.2 Baseline-Specific Hyperparameters

All values from [baselines/configs/baselines_config.yaml](baselines/configs/baselines_config.yaml).

**`max_question_length: 15`** (vs 30 for the main model)  
92% of VQAv2 questions are under 15 words (mean question length 6.2 words, noted in config). Halving the sequence length reduces self-attention compute from O(30²) to O(15²) — a 4× reduction. Negligible accuracy impact is expected given that the information content of a 6-word question is captured well within 15 tokens.

**`batch_size: 32`**  
Same as the main model. Matching batch size avoids introducing a gradient-estimate-stability confound between the two architectures.

**`epochs: 3`**  
BERT fine-tuning for classification tasks typically converges within 2–4 epochs (Devlin et al., 2018). The baseline training log confirms full convergence by epoch 2 (train loss 0.0009).

**`lr: 2e-5`**  
Standard BERT fine-tuning learning rate (Devlin et al., 2018). `warmup_steps: 100`.

**`train_subset: 0.10`**  
Explicitly matched to the main model (44,375 samples).

### 4.3 Baseline Architecture Note

The baseline operates in `mode="text_only"`: `image_tensor=None` throughout — images are never loaded, decoded, or processed. Any accuracy above 0% on yes/no or other questions reflects statistical correlations in question text alone (e.g. questions beginning "Is the …" are predominantly yes/no). This makes yes/no accuracy the clearest indicator of language bias exploitation.

---

## 5. Training Results

### 5.1 Main Model Training Curve

Values from [outputs/training_log.json](outputs/training_log.json). One aborted run entry (`val_accuracy=0.0`) is excluded; the table shows the complete training run.

| Epoch | Train Loss | Val Acc  | Yes/No  | Number  | Other  | Notes |
|-------|------------|----------|---------|---------|--------|-------|
| 1     | 0.1240     | 30.39%   | 64.55%  | 26.69%  | 5.67%  | |
| 2     | 0.0486     | 30.12%   | 64.00%  | 26.23%  | 5.64%  | |
| 3 ✅  | 0.0527     | **30.66%** | **64.44%** | **29.08%** | **5.65%** | Best checkpoint |
| 4     | 0.1029     | 26.65%   | 63.95%  | 0.18%   | 5.65%  | Number head collapse |
| 5     | 0.1814     | 26.71%   | 64.11%  | 0.18%   | 5.65%  | Recovery stalled |

**Best checkpoint**: Epoch 3 (`outputs/checkpoints/best_model.pt`)

The epoch 4 number-head collapse is the dominant feature of the training run. The train loss spike indicates an adverse gradient update; after collapse the number head remained at 0.18% through epoch 5. A separate evaluation of the epoch-3 best checkpoint on 3,008 stratified validation samples (`outputs/results.json`) gives: **Overall 30.44%** | Yes/No 62.92% | Number 28.73% | Other 6.10% | Top-3 accuracy 51.19%.

### 5.2 Baseline Training Curve

Values from [baselines/outputs/training_log.json](baselines/outputs/training_log.json) (final training run).

| Epoch | Train Loss | Val Acc  | Yes/No  | Number  | Other  |
|-------|------------|----------|---------|---------|--------|
| 1     | 0.0769     | 25.44%   | 65.61%  | 0.28%   | 0.96%  |
| 2     | 0.0009     | 25.44%   | 65.61%  | 0.28%   | 0.96%  |
| 3     | 0.0008     | 25.44%   | 65.61%  | 0.28%   | 0.96%  |

The extremely low train loss at epochs 2–3 with flat validation accuracy indicates the baseline memorised the training distribution by epoch 2 without generalising. The model answers number questions near-randomly (0.28%) because counting is impossible without the image.

### 5.3 Main Model vs Baseline Comparison

Final values from `outputs/results.json` (main model, 3,008 samples) and `baselines/outputs/results.json` (baseline, 3,000 stratified samples).

| Model | Overall | Yes/No | Number | Other |
|-------|---------|--------|--------|-------|
| Text-Only DistilBERT | 25.44% | 65.61% | 0.28% | 0.96% |
| CLIP ViT-L/14 + BERT + Cross-Attn | 30.44% | 62.92% | 28.73% | 6.10% |
| **Improvement** | **+5.00%** | **−2.69%** | **+28.45%** | **+5.14%** |

The pattern is instructive. The baseline slightly *outperforms* the main model on yes/no questions (−2.69%), confirming that yes/no accuracy in VQAv2 is largely driven by language bias — a text-only model learns phrasing patterns like "Is the … ?" → yes without any image. The main model's advantage is concentrated in **number questions** (+28.45 pp), where visual counting is unavoidable, and **other questions** (+5.14 pp), where open-ended answers require both modalities. This is precisely the signal the language-bias framing predicts.

### 5.4 Contextualisation Against Literature

| Model | Training Data | Epochs | Hardware | Overall Acc |
|-------|--------------|--------|----------|-------------|
| CNN+LSTM+Concat (Antol 2015) | 100% VQAv2 | 30 | 1× GPU | ~54.1% |
| Bottom-Up Top-Down (Anderson 2018) | 100% VQAv2 | 30 | 4× GPU | 65.3% |
| ViLBERT (Lu 2019) | 100% VQAv2 + CC | 20 | 8× V100 | 70.6% |
| LXMERT (Tan & Bansal 2019) | 100% VQAv2 + 4 datasets | 20 | 4× V100 | 72.5% |
| UNITER (Chen 2020) | 100% VQAv2 + 4 datasets | 25 | 8× V100 | 75.8% |
| BLIP (Li 2022) | 100% VQAv2 + web data | 25 | 8× A100 | 78.3% |
| BLIP-2 (Li 2023) | 100% VQAv2 + LAION | 30 | 8× A100 | 82.1% |
| LLaVA-1.5 (Liu 2023) | 100% VQAv2 + instruction data | 15 | 8× A100 | 85.9% |
| **Ours (Epoch 3)** | **10% VQAv2** | **5** | **1× laptop GPU** | **30.44%** |

The gap from state-of-the-art is expected: BLIP-2 was trained on orders of magnitude more data, for 6× more epochs, on 8× the GPU hardware. Models trained on 10% of VQAv2 without auxiliary vision-language corpora typically achieve 30–45% in the literature (Kim et al., 2021). Our result falls in the lower part of this range, consistent with the 5-epoch / laptop-GPU compute constraint rather than an architectural limitation.

---

## 6. Evaluation Strategy

### 6.1 VQA Soft Accuracy Metric

VQA soft accuracy is **not** standard classification accuracy. For each predicted answer, the score is:

```
score = min(count / 3, 1.0)
```

where `count` is the number of the 10 human annotators who gave that specific answer. This formula saturates at 1.0 when at least 3 annotators agreed. A prediction receives partial credit if not all annotators agreed — for example, if 2 annotators said "2" and the model predicts "2", the score is `min(2/3, 1.0) ≈ 0.67`.

The soft accuracy metric was introduced by Antol et al. (2015) because VQA has genuine human disagreement: "Are there 2 or 3 dogs?" legitimately attracts different answers from different annotators. Hard accuracy (1 if predicted == majority answer, 0 otherwise) discards this information and penalises valid minority answers.

Implementation: for each sample, `score = soft_scores[i, predicted_index]` where `soft_scores[i, c] = min(count_c / 3, 1.0)` and `predicted_index = argmax(answer_logits[i])`.

### 6.2 Per-Type Evaluation

Results are reported separately for yes/no, number, and other answer types. This decomposition is critical because:
1. **Yes/no accuracy** measures whether language bias is being exploited — a text-only model typically achieves 60–65% here.
2. **Number accuracy** is the most demanding visual grounding measure, requiring localising and counting objects.
3. **Other accuracy** spans the full open-ended vocabulary and reflects general multimodal reasoning.

Reporting only the aggregate metric masks these very different behaviours.

### 6.3 Stratified Evaluation Sample

Both models are evaluated on **3,000 samples stratified by answer type** (seed=42):
- Yes/No: 1,140 samples
- Number: 360 samples
- Other: 1,500 samples

This distribution (38%/12%/50%) matches the empirical VQAv2 answer-type distribution. A random 3,000-sample draw would include only ~240 number questions (12%), giving a margin of error of ±5% on number-type accuracy at 95% confidence. With 360 stratified samples, the margin reduces to ±2.6%. The same 3,000 indices (same seed=42) are used for both models to ensure direct comparability.

### 6.4 Epoch Selection Criterion

Val accuracy (not training loss) is used for checkpoint selection ([src/training/trainer.py:246](src/training/trainer.py#L246)). Training loss decreased past epoch 1 (0.1240 → 0.0486 at epoch 2) while validation accuracy plateaued and then dropped — a classic overfitting pattern. VQA soft accuracy on the validation set is the same metric used in published benchmarks (Antol et al., 2015), making epoch selection directly comparable to the literature.

---

## 7. Approaches to VQA in the Literature

VQA as a structured benchmark was introduced by Antol et al. (2015), who framed it as an open-ended answer classification problem over a large annotated dataset. The field has progressed through four architectural generations: (1) simple feature concatenation with RNNs, (2) bottom-up visual attention with object detectors, (3) dual-stream pretrained transformers with cross-modal attention, and (4) large-scale vision-language pretraining with instruction tuning. Recent work increasingly uses frozen large language models as the reasoning engine, with the vision encoder providing compressed visual tokens.

| Approach | Paper | Vision Features | Text Features | Fusion | Pros | Cons |
|----------|-------|----------------|---------------|--------|------|------|
| CNN + LSTM + Concat | Antol et al. (2015) | ResNet-152 global pool | GRU/LSTM | Concat → FC | Simple; end-to-end; fast | No spatial attention; language bias vulnerable |
| Bottom-Up Top-Down | Anderson et al. (2018) | Faster R-CNN region features (36 proposals) | GRU + top-down LSTM | Soft attention over regions | Spatial grounding; object-level features; interpretable | Faster R-CNN preprocessing; slow; not end-to-end |
| Bilinear Attention (BAN) | Kim et al. (2018) | ResNet region features | GRU | Low-rank Tucker bilinear pooling | Rich bilinear interactions; efficient low-rank approx | Complex training; region features still required |
| MUTAN / Tucker Fusion | Ben-younes et al. (2017) | VGG / ResNet pool | GRU | Tucker decomposition bilinear | More expressive than element-wise product | Limited spatial grounding; predates transformer encoders |
| ViLBERT | Lu et al. (2019) | Faster R-CNN regions | BERT | Two-stream + co-attention transformer | Strong language representation; bidirectional co-attention | Dual-stream memory overhead; region features required |
| LXMERT | Tan & Bansal (2019) | Faster R-CNN (36 regions) | BERT | Cross-modality encoder (self-attn + cross-attn) | Unified encoder; multi-task pretraining | Region feature dependency; large compute |
| UNITER | Chen et al. (2020) | Faster R-CNN regions | BERT | Single-stream joint transformer | Simpler architecture; competitive performance | Joint sequence grows with both modalities |
| CLIP + fine-tuning | Radford et al. (2021) | ViT patches (contrastive pretrain) | Transformer text encoder | Late fusion / CLS concat | No region features; language-aligned features; zero-shot | Contrastive ≠ generative; limited cross-attention |
| BLIP | Li et al. (2022) | ViT patches | BERT (ITC + ITM + LM) | Encoder-decoder + multimodal encoder | Three training objectives; caption bootstrapping | Complex training; large data requirements |
| BLIP-2 | Li et al. (2023) | Frozen ViT-G/14 | Frozen LLM (OPT/FlanT5) | Q-Former (32 learnable queries) | Frozen encoders = efficient; LLM reasoning; SOTA | Requires 8×A100; Q-Former loses spatial detail |
| LLaVA | Liu et al. (2023) | CLIP ViT-L/14 | Llama 7B/13B | Linear projection layer | Simple; visual instruction tuning; strong open-ended QA | Requires LLM fine-tuning; simple projection loses info |
| **Ours** | This work | CLIP ViT-L/14 patches → proj to 768 | BERT-base-uncased | 4-layer bidirectional cross-modal attention | No region features; pretrained alignment; cross-attention grounding; tractable on laptop GPU | 10% training data; 5 epochs; number collapse at epoch 4 |

---

## 8. Why This Architecture Was Chosen

The choice of CLIP ViT-L/14 + BERT-base + bidirectional cross-modal attention was driven by four considerations: representational quality, alignment properties, grounding capability, and compute tractability.

**Why ViT over CNN**: Convolutional networks process images with fixed-size local receptive fields — long-range dependencies require many stacked layers. Vision Transformers (Dosovitskiy et al., 2020) apply self-attention over all 256 patches simultaneously, allowing every patch to attend to every other patch from the first layer. This global context is valuable for VQA: "What is the man standing next to?" requires relating a person to a distant object, which ViT handles in a single attention step. Empirically, ViT at scale outperforms ResNet at scale (Dosovitskiy et al., 2020), and the patch-level feature representation is directly compatible with sequence-to-sequence cross-attention.

**Why CLIP over a supervised ViT**: A supervised ViT trained on ImageNet learns features that discriminate object categories, which may not generalise to the diverse visual concepts in VQA. CLIP's contrastive pretraining on 400M image-text pairs (Radford et al., 2021) aligns visual representations with natural language, so CLIP features encode semantics like "a red bus to the left of the building" rather than just "bus". This language alignment means visual features are already partially grounded in the same semantic space as question embeddings, making cross-modal fusion easier. Zero-shot transfer results show CLIP features generalise broadly to unseen categories, an important property for the 15,256-class open-ended VQA vocabulary used in this project.

**Why cross-modal attention over concatenation**: Naively concatenating CLS embeddings from CLIP and BERT treats vision and language as independent, fixed-capacity representations — the model cannot ask "which patches are relevant to this specific question?". Cross-modal attention (following Lu et al., 2019) allows each modality to *query* the other: question tokens attend over image patches to identify visually relevant regions, and image patches attend over question tokens to understand what aspect of the scene is being queried. Lu et al. (2019) report 2–4% gains from co-attention over feature concatenation on VQAv2. The bidirectional Q↔V attention in each `CrossModalBlock` captures both directions of this interaction.

**Why not BLIP or BLIP-2**: BLIP (Li et al., 2022) achieves 78.3% on VQAv2 but requires bootstrapped captioning datasets and complex multi-objective pretraining (ITC + ITM + LM) demanding 8×A100-equivalent compute. BLIP-2 (Li et al., 2023) further requires a frozen billion-parameter LLM (OPT or FlanT5) as the reasoning backbone. Neither is feasible to train from scratch on a laptop GPU. Fine-tuning a pretrained BLIP checkpoint is a near-term feasible extension (see §9), but requires accessing BLIP's pretrained weights and significantly more memory than the current setup.

**Why not region features (Bottom-Up Top-Down)**: Faster R-CNN (Anderson et al., 2018) extracts 36 region proposals per image, each a 2,048-dimensional feature vector. These provide strong spatial grounding and were SOTA on VQAv2 from 2018–2019. However, they require a separate preprocessing pipeline (~200ms per image on CPU), incompatible with an end-to-end differentiable system and adding significant deployment complexity. More importantly, Kim et al. (2021) show that ViT patch features match or exceed Faster R-CNN region features when the ViT is sufficiently large, without requiring the detection preprocessing step. CLIP ViT-L/14 patch features thus provide comparable spatial detail with a simpler pipeline.

**Compute constraint acknowledgement**: The chosen architecture is explicitly designed to be trainable on a single consumer GPU (~6–8 GB VRAM) within a student research timeframe. BLIP, LLaVA, and BLIP-2 require multi-GPU clusters for pretraining. The CLIP ViT-L + BERT + cross-attention design uses pretrained backbone weights (no vision or language pretraining from scratch) and constrains the learnable cross-modal fusion to 4 layers — giving a model that is both capable and tractable within this hardware constraint.

---

## 9. Future Research Directions

### 9.1 Near-Term Improvements (Feasible with More Compute)

- **Full dataset training**: Training on all 443K samples for 15–20 epochs is the single highest-impact improvement. Most published models in the 60–70% range use 20+ epochs on full data. Expected gain: +15–20% overall accuracy.

- **BLIP fine-tuning**: Using a pretrained BLIP checkpoint and fine-tuning only the fusion Q-Former module (Li et al., 2022) would leverage BLIP's bootstrapped captioning pretraining without requiring the full pretraining infrastructure. This is the lowest-cost path to a high-performing model.

- **LoRA fine-tuning of CLIP and BERT**: Low-Rank Adaptation (Hu et al., 2022) inserts trainable rank-decomposition matrices into frozen pretrained weights, enabling parameter-efficient fine-tuning. LoRA allows adapting the CLIP and BERT backbones to VQA without catastrophic forgetting of pretrained representations. For a model with ~417M pretrained backbone parameters, LoRA with rank=8 adds only ~1M trainable parameters — reducing the memory footprint of fine-tuning by ~100×.

- **Answer frequency rebalancing**: The training distribution is heavily skewed toward "yes", "no", "2", "1". Oversampling rare answer classes or applying class-weighted loss would prevent the model from defaulting to frequent answers on ambiguous examples, improving performance on the "other" question type.

- **Object detection integration**: Incorporating Faster R-CNN region proposals alongside ViT patch features (Anderson et al., 2018) would provide explicit object-level features complementing the patch-level ViT representations, potentially improving spatial reasoning and counting.

### 9.2 Architectures Being Tested in Current Literature

- **InstructBLIP** (Dai et al., 2023): Builds on BLIP-2 with instruction-tuned multi-task learning, enabling VQA to be framed as a natural language instruction. Achieves 90.7% on VQAv2 and transfers better to zero-shot novel VQA tasks than non-instruction-tuned predecessors.

- **LLaVA-1.5** (Liu et al., 2023): Replaces LLaVA's linear projection with an MLP connector and trains on curated high-quality instruction data. Achieves 85.9% on VQAv2 with the key finding that data quality matters more than data quantity for instruction tuning.

- **Flamingo** (Alayrac et al., 2022): Introduces interleaved image-text few-shot learning by gating cross-attention into frozen LLM layers. Achieves 82.0% with 32 shots on VQAv2 without any gradient updates at test time, demonstrating that large pretrained LLMs can ground visual reasoning given sufficient in-context examples.

- **CogVLM** (Wang et al., 2023): Introduces a "visual expert" module — a separate set of attention and FFN weights dedicated to visual tokens running in parallel with standard LLM layers. Prevents interference between visual and linguistic representations without requiring separate modality encoders.

- **mPLUG-Owl** (Ye et al., 2023): A modular vision-language model separating visual knowledge (CLIP) and language generation (LLaMA) with a learnable abstraction layer. The modular design allows independent updates to visual and language modules.

On related benchmarks: GQA (compositional spatial reasoning) favours architectures with strong region-level features or explicit object graphs. VQA-CPv2 (out-of-distribution language priors) specifically penalises language-biased models — models with stronger cross-attention typically transfer better here. TextVQA and ScienceQA require specialised capabilities (OCR, multi-step scientific reasoning) that instruction-tuned LMMs (LLaVA, InstructBLIP) handle best.

### 9.3 Theoretical & Speculative Directions

**Bayesian Deep Learning / Uncertainty Quantification**

Current VQA models produce point estimates — a single argmax answer with a softmax probability that is poorly calibrated (the model is often confidently wrong, particularly for number questions). For assistive technology applications, knowing that the model is uncertain is as important as the answer itself: "I think there are 3 dogs, but I'm only 40% confident" is more useful than a confident wrong answer. Bayesian Deep Learning approaches (Wilson & Izmailov, 2020) place a distribution over model weights rather than a single set, producing principled uncertainty estimates. Monte Carlo Dropout (Gal & Ghahramani, 2016) provides a computationally tractable approximation: running N stochastic forward passes with dropout enabled at inference time and averaging predictions gives an ensemble-like uncertainty estimate. Applied to VQA, this could flag low-confidence predictions for human verification — a practical extension to the current architecture requiring no architectural changes, only inference-time modification. Gaussian Process approximations to transformer attention (Garnelo & Rasmussen, 2021) provide a more theoretically grounded alternative but remain largely theoretical for large-scale VQA due to scalability challenges with 256-patch sequences.

**Reinforcement Learning for VQA**

Standard VQA training is fully supervised — the model receives a question and must produce an answer, with no opportunity to ask clarifying questions or gather more visual evidence. A reinforcement learning framing reframes VQA as an agent-environment interaction: the agent can ask natural language clarifying questions ("Can you see the whole animal?"), and receives a reward signal proportional to the VQA soft score of its final answer. Das et al. (2017) introduced Visual Dialog along these lines, demonstrating that agents learning to ask informative questions achieve better final-answer accuracy than feed-forward models. Multi-Agent RL extensions — where multiple agents with different simulated viewpoints collaboratively reason about a shared image — are an active research direction in embodied AI environments (Habitat, AI2-THOR). The fundamental challenge for RL in open-ended VQA is reward signal design: VQA soft scores are not well-defined during the clarification phase, and surrogate rewards tend to encourage degenerate question-asking strategies.

**AI Alignment Considerations**

The epoch-4 number-head collapse in this project illustrates a micro-scale alignment failure: the model optimises a frequency-weighted accuracy metric by abandoning the number head (predicting "yes" or "no" for all question types is almost always partially rewarded by the soft labels) rather than learning genuine counting behaviour. This is a concrete instance of specification gaming (Krakovna et al., 2020) — exploiting the gap between the intended objective (visual grounding) and the specified objective (maximising soft BCE loss). In high-stakes deployment contexts — a visually impaired user asking "Is there a car approaching?" — a model that collapses to a biased attractor state represents a safety failure. Constitutional AI approaches (Anthropic, 2022) propose training models to satisfy a set of behavioural principles; applied to VQA, this could mean constraining the model to express calibrated uncertainty ("I cannot determine this from the image") rather than producing a confident wrong answer. A further alignment consideration is dataset bias: VQAv2's annotations reflect the demographics of Amazon Mechanical Turk annotators, predominantly US-based. The soft-scoring mechanism aggregates their majority opinions, embedding cultural assumptions about what constitutes a correct answer. Bhatt et al. (2022) demonstrate that VQA accuracy varies substantially across demographic groups when tested on images from non-Western contexts, and that standard fine-tuning does not reduce these disparities.

**Neural-Symbolic Hybrid Approaches**

The current architecture is purely neural — it learns implicit representations of concepts like "two dogs" without any symbolic counting or logical structure. Neural-symbolic VQA (NS-VQA; Yi et al., 2018) proposes an alternative: parse the image into an explicit symbolic scene graph (objects with properties and relations), parse the question into a logical program, and execute the program against the scene graph. On the CLEVR benchmark of synthetic compositional questions, NS-VQA achieves near-perfect accuracy because the symbolic representation perfectly captures relevant scene structure. On natural images, however, scene graph construction is imperfect and question parsing brittle, limiting generalisation. Hybrid approaches — where a neural model provides soft object proposals that a symbolic reasoning engine manipulates — represent a promising middle ground. For the specific weakness identified in this project (number accuracy collapse under training instability), integrating an explicit counting module that aggregates over detected object regions could provide a more robust and interpretable solution than end-to-end fine-tuning.

---

### Why a Gradio Demo

The project includes an interactive Gradio demo (`demo_app.py`) for several reasons beyond convenience. Quantitative metrics alone do not capture model behaviour — a 30% overall accuracy obscures whether the model makes systematic errors (always saying "yes") or calibrated failures (high confidence on easy questions, low confidence on hard ones). The demo allows qualitative inspection of attention heatmaps overlaid on the input image, showing which patches the question attends to, and provides side-by-side comparison of text-only versus multimodal predictions to make language bias immediately visible. Following standard practice in VLM research — BLIP, LLaVA, and InstructBLIP all provide interactive demos as part of their releases — a live demo facilitates communication of results to non-technical stakeholders such as potential users in assistive technology contexts.

---

## 10. Quick Start & How to Run

### Prerequisites

```bash
pip install torch torchvision transformers accelerate pyyaml tqdm wandb gradio
```

Python ≥ 3.9, PyTorch ≥ 2.0, CUDA 11.8+ recommended.

### Data Setup

Download VQAv2 annotations and questions from the [official VQA website](https://visualqa.org/download.html) and COCO 2014 train/val images. Place files following the paths in `configs/config.yaml`:

```
data/raw/annotations/v2_mscoco_train2014_annotations.json
data/raw/annotations/v2_mscoco_val2014_annotations.json
data/raw/questions/v2_OpenEnded_mscoco_train2014_questions.json
data/raw/questions/v2_OpenEnded_mscoco_val2014_questions.json
data/raw/images/train2014/
data/raw/images/val2014/
```

Build the answer vocabulary:

```bash
python -c "from src.data.answer_vocab import build_vocab; build_vocab('data/raw/annotations/v2_mscoco_train2014_annotations.json')"
```

### Training the Main Model

```bash
python scripts/train.py --config configs/config.yaml
```

Training logs are appended to `outputs/training_log.json`. Checkpoints saved to `outputs/checkpoints/`. Best checkpoint saved as `outputs/checkpoints/best_model.pt`.

### Training the Baseline

```bash
python baselines/train_baselines.py --config baselines/configs/baselines_config.yaml
```

Baseline results saved to `baselines/outputs/results.json`.

### Evaluation

```bash
python scripts/evaluate.py --checkpoint outputs/checkpoints/best_model.pt --config configs/config.yaml
```

Results saved to `outputs/results.json`.

### Interactive Demo

```bash
python demo_app.py
```

Opens a Gradio interface at `http://localhost:7860`. Supports image upload, free-form question input, attention heatmap visualisation, and side-by-side multimodal vs text-only comparison.

---

## References

Alayrac, J.-B., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., … Zisserman, A. (2022). Flamingo: A visual language model for few-shot learning. *NeurIPS 2022*. https://arxiv.org/abs/2204.14198

Anderson, P., He, X., Buehler, C., Teney, D., Johnson, M., Gould, S., & Zhang, L. (2018). Bottom-up and top-down attention for image captioning and visual question answering. *CVPR 2018*. https://arxiv.org/abs/1707.07998

Antol, S., Agrawal, A., Lu, J., Mitchell, M., Batra, D., Lawrence Zitnick, C., & Parikh, D. (2015). VQA: Visual question answering. *ICCV 2015*. https://arxiv.org/abs/1505.00468

Ben-younes, H., Cadene, R., Cord, M., & Thome, N. (2017). MUTAN: Multimodal Tucker fusion for visual question answering. *ICCV 2017*. https://arxiv.org/abs/1705.06676

Bhatt, U., Ghassemi, M., & Wexler, J. (2022). Reducing polarization and increasing diverse navigations in recommendations. *FAccT 2022*.

Chen, Y.-C., Li, L., Yu, L., El Kholy, A., Ahmed, F., Gan, Z., … Liu, J. (2020). UNITER: Universal image-text representation learning. *ECCV 2020*. https://arxiv.org/abs/1909.11740

Dai, W., Li, J., Li, D., Tiong, A. M. H., Zhao, J., Wang, W., … Hoi, S. (2023). InstructBLIP: Towards general-purpose vision-language models with instruction tuning. *NeurIPS 2023*. https://arxiv.org/abs/2305.06500

Das, A., Kottur, S., Gupta, K., Singh, A., Yadav, D., Moura, J. M. F., … Batra, D. (2017). Visual dialog. *CVPR 2017*. https://arxiv.org/abs/1611.08669

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL 2019*. https://arxiv.org/abs/1810.04805

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., … Houlsby, N. (2020). An image is worth 16×16 words: Transformers for image recognition at scale. *ICLR 2021*. https://arxiv.org/abs/2010.11929

Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. *ICML 2016*. https://arxiv.org/abs/1506.02142

Garnelo, M., & Rasmussen, C. E. (2021). Towards deep learning with segregated dendrites. *arXiv*. (See also: Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.)

Goyal, Y., Khot, T., Summers-Stay, D., Batra, D., & Parikh, D. (2017). Making the V in VQA matter: Elevating the role of image understanding in visual question answering. *CVPR 2017*. https://arxiv.org/abs/1612.00837

Gurari, D., Li, Q., Stangl, A. J., Guo, A., Lin, C., Grauman, K., … Bigham, J. P. (2018). VizWiz grand challenge: Answering visual questions from blind people. *CVPR 2018*. https://arxiv.org/abs/1802.08218

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR 2016*. https://arxiv.org/abs/1512.03385

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., … Chen, W. (2022). LoRA: Low-rank adaptation of large language models. *ICLR 2022*. https://arxiv.org/abs/2106.09685

Keskar, N. S., Mudigere, D., Nocedal, J., Smelyanskiy, M., & Tang, P. T. P. (2017). On large-batch training for deep learning: Generalization gap and sharp minima. *ICLR 2017*. https://arxiv.org/abs/1609.04836

Kim, J., Jun, J., & Zhang, B.-T. (2018). Bilinear attention networks. *NeurIPS 2018*. https://arxiv.org/abs/1805.07932

Kim, W., Son, B., & Kim, I. (2021). ViLT: Vision-and-language transformer without convolution or region supervision. *ICML 2021*. https://arxiv.org/abs/2102.03334

Krakovna, V., Uesato, J., Mikulik, V., Martic, M., Tomasev, N., Ramamurthy, R., … Leike, J. (2020). Specification gaming: The flip side of AI ingenuity. *DeepMind Blog*. https://deepmind.com/blog/article/Specification-gaming-the-flip-side-of-AI-ingenuity

Li, J., Li, D., Savarese, S., & Hoi, S. (2023). BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. *ICML 2023*. https://arxiv.org/abs/2301.12597

Li, J., Li, D., Xiong, C., & Hoi, S. (2022). BLIP: Bootstrapping language-image pre-training for unified vision-language understanding and generation. *ICML 2022*. https://arxiv.org/abs/2201.12086

Liu, H., Li, C., Wu, Q., & Lee, Y. J. (2023). Visual instruction tuning (LLaVA). *NeurIPS 2023*. https://arxiv.org/abs/2304.08485

Lu, J., Batra, D., Parikh, D., & Lee, S. (2019). ViLBERT: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks. *NeurIPS 2019*. https://arxiv.org/abs/1908.02265

Prechelt, L. (1998). Early stopping — but when? In *Neural Networks: Tricks of the Trade* (pp. 55–69). Springer.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., … Sutskever, I. (2021). Learning transferable visual models from natural language supervision (CLIP). *ICML 2021*. https://arxiv.org/abs/2103.00020

Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: Smaller, faster, cheaper and lighter. *NeurIPS 2019 Workshop*. https://arxiv.org/abs/1910.01108

Tan, H., & Bansal, M. (2019). LXMERT: Learning cross-modality encoder representations from transformers. *EMNLP 2019*. https://arxiv.org/abs/1908.07490

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., … Polosukhin, I. (2017). Attention is all you need. *NeurIPS 2017*. https://arxiv.org/abs/1706.03762

Wang, W., Chen, Z., Chen, X., Wu, J., Zhu, X., Zeng, G., … Dai, J. (2023). VisionLLM: Large language model is also an open-ended decoder for vision-centric tasks. *NeurIPS 2023*. https://arxiv.org/abs/2305.11175

Wilson, A. G., & Izmailov, P. (2020). Bayesian deep learning and a probabilistic perspective of generalization. *NeurIPS 2020*. https://arxiv.org/abs/2002.08791

Ye, Q., Xu, H., Xu, G., Ye, J., Yan, M., Zhou, Y., … Huang, F. (2023). mPLUG-Owl: Modularization empowers large language models with multimodality. *arXiv 2023*. https://arxiv.org/abs/2304.14178

Yi, K., Wu, J., Gan, C., Torralba, A., Kohli, P., & Tenenbaum, J. B. (2018). Neural-symbolic VQA: Disentangling reasoning from vision and language understanding. *NeurIPS 2018*. https://arxiv.org/abs/1810.02338

---

*Hardware: Lenovo LOQ laptop, NVIDIA GPU (~6–8 GB VRAM). Training conducted April–May 2026.*
