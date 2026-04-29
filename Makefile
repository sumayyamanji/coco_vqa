.PHONY: setup vocab train train-all evaluate demo demo-inference test docker clean

setup:
	bash setup.sh

vocab:
	python scripts/build_vocab.py

train:
	python scripts/train.py --config configs/config.yaml --mode multimodal

train-all:
	python scripts/train.py --config configs/config.yaml --mode multimodal
	python scripts/train.py --config configs/config.yaml --mode text_only
	python scripts/train.py --config configs/config.yaml --mode image_only

evaluate:
	python scripts/evaluate.py --config configs/config.yaml

demo:
	python demo/app.py

demo-inference:
	python scripts/demo_inference.py --image $(IMAGE) --question "$(QUESTION)" $(ARGS)

test:
	pytest tests/ -v

docker:
	docker build -t coco-vqa .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} +
	rm -f outputs/eval_plots/*
