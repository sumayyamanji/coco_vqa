"""Gradio demo — tabbed VQA interface: ask, compare, and example gallery."""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml
from PIL import Image, ImageDraw

import gradio as gr

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)

# ── Optional HuggingFace Spaces GPU decorator ────────────────────────────────
try:
    import spaces as _spaces
    def _gpu(fn):
        return _spaces.GPU(fn)
except ImportError:
    def _gpu(fn):
        return fn

# ── Project imports ───────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.utils import ROOT_DIR, setup_output_dirs
    from src.data.augmentations import get_val_transforms
    from src.data.answer_vocab import AnswerVocab
    from src.models.vqa_model import VQAModel
    from src.utils.checkpoint import load_checkpoint
    from src.evaluation.visualisation import (
        plot_attention_heatmap,
        plot_visual_grounding,
        plot_scene_graph,
    )
    _SRC_OK = True
except Exception as _err:
    _SRC_OK = False
    ROOT_DIR = Path(__file__).parent.parent
    _log.warning("Could not import src modules: %s", _err)

# ── Config ────────────────────────────────────────────────────────────────────
_ROOT = ROOT_DIR
_CFG_PATH = _ROOT / "configs" / "config.yaml"
with open(_CFG_PATH) as _fh:
    cfg: dict = yaml.safe_load(_fh)

if _SRC_OK:
    try:
        setup_output_dirs(cfg)
    except Exception:
        pass

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODE_MAP = {"Multimodal": "multimodal", "Text Only": "text_only", "Image Only": "image_only"}
_TYPE_NAMES = {0: "yes/no", 1: "number", 2: "other"}
_NO_MODEL_MSG = (
    "⚠️ No trained model found. "
    "Please train first (`python scripts/train.py`) or download a checkpoint to `checkpoints/`."
)
_NO_VOCAB_MSG = (
    "⚠️ No answer vocab found. "
    "Please run `python scripts/build_vocab.py` first."
)

# ── Vocab ─────────────────────────────────────────────────────────────────────
_vocab: Optional[Any] = None
_idx2ans: Dict[int, str] = {}

if _SRC_OK:
    _vpath = _ROOT / cfg["paths"]["vocab_path"]
    if _vpath.exists():
        try:
            _vocab = AnswerVocab.load(_vpath)
            _idx2ans = {i: _vocab.idx_to_answer(i) for i in range(len(_vocab))}
            _log.info("Vocab loaded: %d answers.", len(_vocab))
        except Exception as _e:
            _log.warning("Could not load vocab: %s", _e)

# ── Transform + tokeniser ─────────────────────────────────────────────────────
_transform = None
_tokenizer = None

if _SRC_OK:
    try:
        from transformers import BertTokenizer
        _transform = get_val_transforms(cfg["data"]["image_size"])
        _tokenizer = BertTokenizer.from_pretrained(cfg["model"]["text_encoder"])
    except Exception as _e:
        _log.warning("Could not load transform/tokeniser: %s", _e)

# ── Models (one per mode) ─────────────────────────────────────────────────────
_models: Dict[str, Optional[Any]] = {"multimodal": None, "text_only": None, "image_only": None}


def _find_checkpoint(mode: str) -> Optional[Path]:
    ckpt_dir = _ROOT / cfg["paths"]["checkpoint_dir"]
    if not ckpt_dir.exists():
        return None
    suffix_map = {"text_only": "_text", "image_only": "_image", "multimodal": ""}
    specific = ckpt_dir / f"best_model{suffix_map[mode]}.pt"
    if specific.exists():
        return specific
    generic = ckpt_dir / "best_model.pt"
    if generic.exists():
        return generic
    latest = sorted(ckpt_dir.glob("checkpoint_epoch*.pt"))
    return latest[-1] if latest else None


def _load_model(mode: str) -> Optional[Any]:
    if not _SRC_OK or _vocab is None:
        return None
    ckpt = _find_checkpoint(mode)
    if ckpt is None:
        _log.info("No checkpoint found for mode '%s'.", mode)
        return None
    try:
        m = VQAModel(cfg, vocab_size=len(_vocab), mode=mode).to(_DEVICE)
        load_checkpoint(ckpt, m, device=_DEVICE)
        m.eval()
        _log.info("Loaded %s model from %s.", mode, ckpt.name)
        return m
    except Exception as _e:
        _log.warning("Failed to load %s model: %s", mode, _e)
        return None


if _SRC_OK:
    for _mode_key in list(_models):
        _models[_mode_key] = _load_model(_mode_key)

# ── RAG retriever (optional) ──────────────────────────────────────────────────
_retriever = None
_ret_cfg = cfg.get("retrieval", {})

if _SRC_OK and _ret_cfg.get("use_rag", False):
    _idx_dir = _ROOT / cfg["paths"].get("faiss_index", _ret_cfg.get("faiss_index_path", "data/faiss_index"))
    if (_idx_dir / "index.faiss").exists():
        try:
            from src.retrieval import CLIPEmbedder, VQAIndex, RAGRetriever
            _emb = CLIPEmbedder(cfg["model"]["vision_backbone"], device=str(_DEVICE))
            _vidx = VQAIndex.load(_idx_dir)
            _retriever = RAGRetriever(_vidx, _emb)
            _log.info("RAG retriever loaded (%d entries).", _vidx.size)
        except Exception as _e:
            _log.warning("Could not load RAG retriever: %s", _e)

_rag_ready = _retriever is not None

# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fig_to_pil(fig: plt.Figure) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    out = Image.open(buf).copy()
    buf.close()
    plt.close(fig)
    return out


def _make_batch(
    pil_image: Optional[Image.Image],
    question: str,
    mode: str,
) -> Dict[str, Any]:
    batch: Dict[str, Any] = {}
    if mode != "text_only" and pil_image is not None:
        batch["image_tensor"] = (
            _transform(pil_image.convert("RGB")).unsqueeze(0).to(_DEVICE)
        )
    else:
        batch["image_tensor"] = None

    if mode != "image_only" and question.strip() and _tokenizer is not None:
        enc = _tokenizer(
            question,
            padding="max_length",
            truncation=True,
            max_length=cfg["data"]["max_question_length"],
            return_tensors="pt",
        )
        batch["question_ids"] = enc["input_ids"].to(_DEVICE)
        batch["attention_mask"] = enc["attention_mask"].to(_DEVICE)
    else:
        batch["question_ids"] = None
        batch["attention_mask"] = None

    return batch


def _decode_prediction(out: Dict[str, Any]) -> Tuple[str, str, List[List]]:
    """Return (answer, type_name, top3_rows) from raw model output."""
    type_idx = int(out["answer_type_logits"].argmax(dim=-1)[0].item())
    type_name = _TYPE_NAMES.get(type_idx, "other")

    if type_idx == 0:
        yn = int(out["yes_no_logits"].argmax(dim=-1)[0].item())
        answer = "yes" if yn == 1 else "no"
    elif type_idx == 1:
        answer = str(int(out["number_logits"].argmax(dim=-1)[0].item()))
    else:
        top_idx = int(out["top3_answers"]["indices"][0][0].item())
        answer = _idx2ans.get(top_idx, f"<{top_idx}>")

    top3_rows: List[List] = []
    for j in range(3):
        idx = int(out["top3_answers"]["indices"][0][j].item())
        prob = float(out["top3_answers"]["probs"][0][j].item())
        top3_rows.append([_idx2ans.get(idx, f"<{idx}>"), round(prob, 4)])

    return answer, type_name, top3_rows


def _run_inference(
    pil_image: Optional[Image.Image],
    question: str,
    mode_key: str,
    use_rag: bool,
) -> Dict[str, Any]:
    model = _models.get(mode_key)
    if model is None:
        return {"error": _NO_MODEL_MSG}
    if _vocab is None:
        return {"error": _NO_VOCAB_MSG}

    effective_q = question
    if use_rag and _retriever is not None and pil_image is not None:
        k = _ret_cfg.get("top_k_retrieval", 3)
        effective_q = _retriever.augment_question(pil_image, question, k=k)

    batch = _make_batch(pil_image, effective_q, mode_key)
    with torch.no_grad():
        out = model(batch)

    answer, type_name, top3 = _decode_prediction(out)
    confidence = float(out["confidence"][0].item())

    return {
        "answer": answer,
        "type": type_name,
        "top3": top3,
        "confidence": confidence,
        "attn": out.get("cross_attention_weights"),
        "sg": out.get("scene_graph_output"),
        "img_tensor": batch.get("image_tensor"),
    }


def _try_heatmap(
    pil_image: Optional[Image.Image],
    result: Dict[str, Any],
    question: str,
) -> Optional[Image.Image]:
    attn = result.get("attn")
    if attn is None or pil_image is None:
        return None
    try:
        fig = plot_attention_heatmap(pil_image, attn[0], question, result["answer"])
        return _fig_to_pil(fig)
    except Exception:
        return None


def _try_grounding(
    pil_image: Optional[Image.Image],
    result: Dict[str, Any],
    question: str,
) -> Optional[Image.Image]:
    attn = result.get("attn")
    if attn is None or pil_image is None:
        return None
    try:
        fig = plot_visual_grounding(pil_image, attn[0], question, result["answer"])
        return _fig_to_pil(fig)
    except Exception:
        return None


def _try_scene_graph(result: Dict[str, Any]) -> Optional[Image.Image]:
    sg = result.get("sg")
    if sg is None:
        return None
    try:
        fig = plot_scene_graph(sg[0])
        return _fig_to_pil(fig)
    except Exception:
        return None


def _confidence_bar_chart(confs: Dict[str, float], best: str) -> Image.Image:
    labels = list(confs.keys())
    vals = list(confs.values())
    colors = ["#2ecc71" if lbl == best else "#3498db" for lbl in labels]

    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.bar(labels, vals, color=colors, edgecolor="white", width=0.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Confidence")
    ax.set_title("Confidence by Mode  (green = highest)")
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(v + 0.03, 0.95),
            f"{v:.1%}",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )
    fig.tight_layout()
    return _fig_to_pil(fig)

# ─────────────────────────────────────────────────────────────────────────────
# Gradio callback functions
# ─────────────────────────────────────────────────────────────────────────────

_EMPTY_TOP3 = [["—", 0.0], ["—", 0.0], ["—", 0.0]]


@_gpu
def ask_fn(
    pil_image: Optional[Image.Image],
    question: str,
    mode_label: str,
    use_rag: bool,
    history: List[List],
) -> Tuple:
    """Tab 1 — single-mode inference."""
    def _bail(msg: str):
        return msg, "—", _EMPTY_TOP3, "—", None, None, None, history, history

    if pil_image is None:
        return _bail("⚠️ Please upload an image.")
    if not question.strip():
        return _bail("⚠️ Please enter a question.")

    mode_key = _MODE_MAP.get(mode_label, "multimodal")
    result = _run_inference(pil_image, question, mode_key, use_rag)

    if "error" in result:
        return _bail(result["error"])

    answer = result["answer"]
    conf_str = f"{result['confidence']:.1%}"
    top3 = result["top3"]
    type_str = result["type"]

    heatmap_img = _try_heatmap(pil_image, result, question)
    grounding_img = _try_grounding(pil_image, result, question)
    scene_img = _try_scene_graph(result)

    updated = history + [[question, answer, conf_str, mode_label]]

    return (
        answer, conf_str, top3, type_str,
        heatmap_img, grounding_img, scene_img,
        updated, updated,
    )


@_gpu
def compare_fn(
    pil_image: Optional[Image.Image],
    question: str,
) -> Tuple:
    """Tab 2 — run all three modes and return side-by-side results."""
    def _bail(msg: str):
        err4 = (msg, "—", _EMPTY_TOP3, None)
        err3 = (msg, "—", _EMPTY_TOP3)
        return (*err4, *err3, *err4, "—", None)

    if pil_image is None:
        return _bail("⚠️ Please upload an image.")
    if not question.strip():
        return _bail("⚠️ Please enter a question.")

    raw: Dict[str, Dict] = {
        lbl: _run_inference(pil_image, question, key, False)
        for lbl, key in _MODE_MAP.items()
    }

    def _pack4(lbl: str) -> Tuple:
        r = raw[lbl]
        if "error" in r:
            return r["error"], "—", _EMPTY_TOP3, None
        return (
            r["answer"],
            f"{r['confidence']:.1%}",
            r["top3"],
            _try_heatmap(pil_image, r, question),
        )

    mm = _pack4("Multimodal")
    to_ = _pack4("Text Only")
    io_ = _pack4("Image Only")

    confs = {
        lbl: (raw[lbl].get("confidence", 0.0) if "error" not in raw[lbl] else 0.0)
        for lbl in _MODE_MAP
    }
    best = max(confs, key=confs.get)
    winner_md = f"**🏆 Highest confidence: {best}** ({confs[best]:.1%})"
    chart = _confidence_bar_chart(confs, best)

    # outputs: mm(4) + to_(3, no heatmap) + io_(4) + summary(2) = 13
    return (*mm, *to_[:3], *io_, winner_md, chart)


@_gpu
def run_example_fn(
    pil_image: Optional[Image.Image],
    question: str,
) -> Tuple[str, str, str, str]:
    """Tab 3 — text-only and multimodal answers for the selected example."""
    if pil_image is None or not question.strip():
        return "—", "—", "—", "—"

    to_r = _run_inference(pil_image, question, "text_only", False)
    mm_r = _run_inference(pil_image, question, "multimodal", False)

    to_ans = to_r.get("answer", to_r.get("error", "—"))
    to_conf = f"{to_r['confidence']:.1%}" if "confidence" in to_r else "—"
    mm_ans = mm_r.get("answer", mm_r.get("error", "—"))
    mm_conf = f"{mm_r['confidence']:.1%}" if "confidence" in mm_r else "—"

    return to_ans, to_conf, mm_ans, mm_conf

# ─────────────────────────────────────────────────────────────────────────────
# Placeholder example images (created once at startup if absent)
# ─────────────────────────────────────────────────────────────────────────────

_EXAMPLES_DIR = _ROOT / "demo" / "examples"
_EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

_EXAMPLE_SPECS = [
    ("counting.jpg",  "How many shapes are in the image?"),
    ("color.jpg",     "What colour is the large rectangle?"),
    ("yesno.jpg",     "Is there a green shape in the image?"),
    ("spatial.jpg",   "What colour is on the left side?"),
    ("object.jpg",    "What shape is in the centre of the image?"),
    ("scene.jpg",     "Is the sky blue in this image?"),
]


def _draw_example(fname: str, path: Path) -> None:
    """Create a simple placeholder PIL image for each example category."""
    img = Image.new("RGB", (224, 224), "white")
    draw = ImageDraw.Draw(img)

    if fname == "counting.jpg":
        img = Image.new("RGB", (224, 224), (220, 205, 185))
        draw = ImageDraw.Draw(img)
        draw.ellipse([25, 65, 85, 125], fill="#e74c3c")
        draw.ellipse([95, 30, 155, 90], fill="#3498db")
        draw.ellipse([140, 110, 200, 170], fill="#2ecc71")

    elif fname == "color.jpg":
        img = Image.new("RGB", (224, 224), (245, 248, 252))
        draw = ImageDraw.Draw(img)
        draw.rectangle([42, 72, 182, 152], fill="#3498db")

    elif fname == "yesno.jpg":
        img = Image.new("RGB", (224, 224), (235, 252, 235))
        draw = ImageDraw.Draw(img)
        draw.ellipse([57, 57, 167, 167], fill="#27ae60")

    elif fname == "spatial.jpg":
        img = Image.new("RGB", (224, 224), "white")
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, 112, 224], fill="#e74c3c")
        draw.rectangle([112, 0, 224, 224], fill="#3498db")

    elif fname == "object.jpg":
        img = Image.new("RGB", (224, 224), (255, 255, 220))
        draw = ImageDraw.Draw(img)
        draw.ellipse([77, 77, 147, 147], fill="#f39c12")

    elif fname == "scene.jpg":
        img = Image.new("RGB", (224, 224), (135, 206, 235))
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 155, 224, 224], fill="#27ae60")
        draw.ellipse([148, 28, 208, 88], fill=(255, 255, 200))

    img.save(path)


def _ensure_example_images() -> List[List[str]]:
    rows: List[List[str]] = []
    for fname, question in _EXAMPLE_SPECS:
        fpath = _EXAMPLES_DIR / fname
        if not fpath.exists():
            _draw_example(fname, fpath)
        rows.append([str(fpath), question])
    return rows


_EXAMPLE_DATA = _ensure_example_images()

# ─────────────────────────────────────────────────────────────────────────────
# Gradio Blocks layout
# ─────────────────────────────────────────────────────────────────────────────

_TITLE = "# 🧠 Visual Question Answering — Multimodal vs Text-Only"
_DESCRIPTION = (
    "Upload any image, ask a question in natural language, and see how well the model "
    "answers with **vision + language** (Multimodal), **language alone** (Text Only), "
    "and **vision alone** (Image Only).  Trained on VQA v2 with a CLIP + BERT backbone."
)
_FOOTER = (
    "---\n"
    "**Links:** "
    "[GitHub](https://github.com) &nbsp;·&nbsp; "
    "[HuggingFace Hub](https://huggingface.co) &nbsp;·&nbsp; "
    "Built with [Gradio](https://gradio.app) & PyTorch"
)
_rag_label = (
    "Use RAG context"
    if _rag_ready
    else "Use RAG context *(index not built — run `build_vocab.py --build-index`)*"
)

with gr.Blocks(theme=gr.themes.Soft(), title="VQA Demo") as demo:

    gr.Markdown(_TITLE)
    gr.Markdown(_DESCRIPTION)

    # ── Tab 1: Ask a Question ─────────────────────────────────────────────────
    with gr.Tab("🔍 Ask a Question"):
        history_state = gr.State([])

        with gr.Row():
            # Left: inputs
            with gr.Column(scale=1, min_width=320):
                img1 = gr.Image(type="pil", label="Upload Image")
                q1 = gr.Textbox(
                    placeholder="Ask anything about the image…",
                    label="Your Question",
                    lines=2,
                )
                mode1 = gr.Radio(
                    choices=["Multimodal", "Text Only", "Image Only"],
                    value="Multimodal",
                    label="Mode",
                )
                rag1 = gr.Checkbox(
                    label=_rag_label,
                    value=False,
                    interactive=_rag_ready,
                )
                ask_btn = gr.Button("Ask", variant="primary")

            # Right: outputs
            with gr.Column(scale=1, min_width=320):
                answer1    = gr.Textbox(label="Answer", interactive=False)
                conf1      = gr.Textbox(label="Confidence", interactive=False)
                top3_1     = gr.Dataframe(
                    headers=["Answer", "Probability"],
                    datatype=["str", "number"],
                    label="Top-3 Answers",
                    interactive=False,
                    row_count=3,
                )
                type1      = gr.Textbox(label="Answer Type", interactive=False)
                heatmap1   = gr.Image(label="Attention Heatmap", interactive=False)
                grounding1 = gr.Image(label="Visual Grounding", interactive=False)
                scene1     = gr.Image(label="Scene Graph", interactive=False)

        with gr.Row():
            history_df = gr.Dataframe(
                headers=["Question", "Answer", "Confidence", "Mode"],
                label="Question History (this session)",
                interactive=False,
            )

        _ask_inputs  = [img1, q1, mode1, rag1, history_state]
        _ask_outputs = [
            answer1, conf1, top3_1, type1,
            heatmap1, grounding1, scene1,
            history_df, history_state,
        ]
        ask_btn.click(fn=ask_fn, inputs=_ask_inputs, outputs=_ask_outputs)
        q1.submit(fn=ask_fn, inputs=_ask_inputs, outputs=_ask_outputs)

    # ── Tab 2: Side-by-Side Comparison ───────────────────────────────────────
    with gr.Tab("⚔️ Side-by-Side Comparison"):

        with gr.Row():
            img2 = gr.Image(type="pil", label="Upload Image", scale=1)
            with gr.Column(scale=1):
                q2 = gr.Textbox(
                    placeholder="Ask anything about the image…",
                    label="Your Question",
                    lines=2,
                )
                compare_btn = gr.Button("Compare All Modes", variant="primary")

        gr.Markdown("---")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🌐 Multimodal")
                answer_mm  = gr.Textbox(label="Answer",      interactive=False)
                conf_mm    = gr.Textbox(label="Confidence",  interactive=False)
                top3_mm    = gr.Dataframe(
                    headers=["Answer", "Probability"],
                    datatype=["str", "number"],
                    interactive=False, row_count=3,
                )
                heatmap_mm = gr.Image(label="Attention Heatmap", interactive=False)

            with gr.Column():
                gr.Markdown("### 📝 Text Only")
                answer_to  = gr.Textbox(label="Answer",      interactive=False)
                conf_to    = gr.Textbox(label="Confidence",  interactive=False)
                top3_to    = gr.Dataframe(
                    headers=["Answer", "Probability"],
                    datatype=["str", "number"],
                    interactive=False, row_count=3,
                )
                gr.Markdown("*No visual attention in text-only mode.*")

            with gr.Column():
                gr.Markdown("### 🖼️ Image Only")
                answer_io  = gr.Textbox(label="Answer",      interactive=False)
                conf_io    = gr.Textbox(label="Confidence",  interactive=False)
                top3_io    = gr.Dataframe(
                    headers=["Answer", "Probability"],
                    datatype=["str", "number"],
                    interactive=False, row_count=3,
                )
                heatmap_io = gr.Image(label="Attention Heatmap", interactive=False)

        with gr.Row():
            winner_label = gr.Markdown("")
        with gr.Row():
            conf_chart = gr.Image(
                label="Confidence Comparison", interactive=False
            )

        _cmp_inputs  = [img2, q2]
        _cmp_outputs = [
            answer_mm, conf_mm, top3_mm, heatmap_mm,   # 4
            answer_to, conf_to, top3_to,                # 3
            answer_io, conf_io, top3_io, heatmap_io,    # 4
            winner_label, conf_chart,                   # 2  → total 13
        ]
        compare_btn.click(fn=compare_fn, inputs=_cmp_inputs, outputs=_cmp_outputs)
        q2.submit(fn=compare_fn, inputs=_cmp_inputs, outputs=_cmp_outputs)

    # ── Tab 3: Examples ───────────────────────────────────────────────────────
    with gr.Tab("📊 Examples"):
        gr.Markdown(
            "Click a row below to load it, then press **Run Example** "
            "to see both Text-Only and Multimodal answers side by side."
        )

        with gr.Row():
            ex_img = gr.Image(
                type="pil", label="Example Image",
                interactive=False, height=224,
            )
            with gr.Column():
                ex_q = gr.Textbox(label="Question", interactive=False, lines=2)
                run_ex_btn = gr.Button("Run Example", variant="primary")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📝 Text-Only Answer")
                ex_ans_to  = gr.Textbox(label="Answer",     interactive=False)
                ex_conf_to = gr.Textbox(label="Confidence", interactive=False)
            with gr.Column():
                gr.Markdown("### 🌐 Multimodal Answer")
                ex_ans_mm  = gr.Textbox(label="Answer",     interactive=False)
                ex_conf_mm = gr.Textbox(label="Confidence", interactive=False)

        gr.Examples(
            examples=_EXAMPLE_DATA,
            inputs=[ex_img, ex_q],
            label="Example Questions — click any row to load",
            examples_per_page=6,
        )

        run_ex_btn.click(
            fn=run_example_fn,
            inputs=[ex_img, ex_q],
            outputs=[ex_ans_to, ex_conf_to, ex_ans_mm, ex_conf_mm],
        )

    # ── Footer ────────────────────────────────────────────────────────────────
    gr.Markdown(_FOOTER)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
