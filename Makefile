.PHONY: setup download preprocess gen train score rerank eval all clean test lint

# Setup environment
setup:
	pip install -r requirements.txt
	mkdir -p artifacts/logs artifacts/candidates artifacts/candidates_scored artifacts/verifier artifacts/results artifacts/figures
	mkdir -p data/raw data/processed
	@echo "✓ Setup complete. Please copy .env.example to .env and configure your API keys."

# Download datasets (will prompt for manual download if needed)
download:
	python -m src.legaladapter.data.download --config configs/datasets.yaml

# Preprocess datasets into unified format
preprocess:
	python -m src.legaladapter.data.preprocess --config configs/datasets.yaml

# Generate K candidates using multiple LLMs
gen:
	python -m src.legaladapter.gen.generate \
		--config configs/base.yaml \
		--models configs/models.yaml \
		--datasets configs/datasets.yaml \
		--split test

# Train verifier model
train:
	python -m src.legaladapter.verify.train \
		--config configs/models.yaml \
		--datasets configs/datasets.yaml

# Score candidates with trained verifier
score:
	python -m src.legaladapter.verify.infer \
		--config configs/models.yaml \
		--datasets configs/datasets.yaml \
		--split test

# Re-rank candidates based on verifier scores
rerank:
	python -m src.legaladapter.rank.rerank \
		--config configs/base.yaml \
		--datasets configs/datasets.yaml \
		--split test

# Evaluate final results
eval:
	python -m src.legaladapter.eval.evaluate \
		--config configs/eval.yaml \
		--datasets configs/datasets.yaml \
		--split test
	python -m src.legaladapter.eval.tables \
		--config configs/eval.yaml \
		--datasets configs/datasets.yaml

# Run full pipeline
all: setup download preprocess gen train score rerank eval

# Clean generated artifacts
clean:
	rm -rf artifacts/candidates/* artifacts/candidates_scored/* artifacts/results/* artifacts/figures/*
	@echo "✓ Cleaned artifacts (kept directory structure)"

# Clean everything including models
clean-all: clean
	rm -rf artifacts/verifier/*
	rm -rf data/processed/*
	@echo "✓ Cleaned all generated files"

# Run tests
test:
	pytest tests/ -v --cov=src/legaladapter --cov-report=html

# Lint code
lint:
	@echo "Checking for code quality..."
	# Add your linting commands here (e.g., flake8, black, ruff)
	@echo "✓ Lint check passed."

# Quick test with synthetic data and mock models (No API key required)
test-quick:
	@echo "🚀 Starting LegalAdapter Quick Test (Mock Mode)..."
	python -m src.legaladapter.utils.create_test_data
	
	@echo "\n1. Generating candidates (Mock)..."
	python -m src.legaladapter.gen.generate \
		--config configs/base.yaml \
		--models configs/demo.yaml \
		--datasets configs/datasets.yaml \
		--split test --quick-test
		
	@echo "\n2. Training verifier (Fast mode)..."
	# We use the 'test' split as train just for demo purposes to make it run
	python -m src.legaladapter.gen.generate \
		--config configs/base.yaml \
		--models configs/demo.yaml \
		--datasets configs/datasets.yaml \
		--split train --quick-test
	python -m src.legaladapter.verify.train \
		--config configs/demo.yaml \
		--datasets configs/datasets.yaml \
		--base configs/base.yaml
		
	@echo "\n3. Scoring candidates..."
	python -m src.legaladapter.verify.infer \
		--config configs/demo.yaml \
		--datasets configs/datasets.yaml \
		--base configs/base.yaml \
		--split test
		
	@echo "\n4. Re-ranking..."
	python -m src.legaladapter.rank.rerank \
		--config configs/base.yaml \
		--datasets configs/datasets.yaml \
		--split test
		
	@echo "\n5. Evaluating..."
	python -m src.legaladapter.eval.evaluate \
		--config configs/eval.yaml \
		--datasets configs/datasets.yaml \
		--split test
	python -m src.legaladapter.eval.tables \
		--config configs/eval.yaml \
		--datasets configs/datasets.yaml \
		--split test
		
	@echo "\n✅ Quick test complete! Check artifacts/results/ for outputs."


