LegalAdapter

LegalAdapter implements a K-candidate generation + verifier re-ranking framework for legal question answering. It generates multiple candidate answers using diverse LLMs and selects the best one via a lightweight verifier.

Prerequisites

Python 3.8+

GPU optional (recommended 4GB+ VRAM)

Optional API keys: OpenAI, xAI

Installation
pip install -r requirements.txt

Configuration
Device Setup

Edit configs/base.yaml:

device: "cuda"  # or "cpu"

API Keys

Copy .env.example to .env and add your keys:

OPENAI_API_KEY=sk-...
XAI_API_KEY=...

Model Settings

Edit configs/models.yaml:

generators:
  openai_gpt4o:
    enabled: true
  xai_grok2:
    enabled: false
  ollama_mistral:
    enabled: false

Verifier Model
verifier:
  backbone: "microsoft/deberta-v3-base"
  batch_size: 32

Quick Start
Step 1: Create test data
python -m src.legaladapter.utils.create_test_data

Step 2: Run full pipeline
make test-quick

Step-by-Step Usage
Generate candidates
python -m src.legaladapter.gen.generate --split test

Train verifier
python -m src.legaladapter.verify.train

Score and rerank
python -m src.legaladapter.verify.infer --split test
python -m src.legaladapter.rank.rerank --split test

Evaluate
python -m src.legaladapter.eval.evaluate --split test

Outputs

Results are saved in artifacts/:

artifacts/
  results/
    results_table_test.csv
    coliee_task4_test_predictions.json
  verifier/
    best_model.pt

License

MIT License. See LICENSE for details.

artifacts/verifier/best_model.pt

License

MIT License. See LICENSE for details.
