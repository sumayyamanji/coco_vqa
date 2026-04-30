#!/usr/bin/env bash
# setup.sh — create a local Python 3.10 venv and install all dependencies
set -euo pipefail

VENV_DIR="coco_vqa_env"

echo "==> Creating virtual environment at ./${VENV_DIR}/"
python -m venv "${VENV_DIR}"

echo "==> Activating virtual environment"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "==> Upgrading pip"
pip install --upgrade pip

echo "==> Installing requirements"
pip install -r requirements.txt

echo "==> Downloading spaCy model en_core_web_sm"
python -m spacy download en_core_web_sm

echo "==> Creating output directories"
mkdir -p outputs/checkpoints
mkdir -p outputs/eval_plots
mkdir -p outputs/eda_plots
mkdir -p outputs/heatmaps
mkdir -p outputs/scene_graphs
mkdir -p data/faiss_index
mkdir -p data/raw/annotations data/raw/questions data/raw/images
mkdir -p data/processed

echo ""
echo "======================================================"
echo "Setup complete."
echo ""
echo "Run:  source ${VENV_DIR}/bin/activate"
echo "Then: python scripts/build_vocab.py"
echo "Then: python scripts/train.py --config configs/config.yaml --mode multimodal"
echo "======================================================"
