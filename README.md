# LegalAdapter: Reliable Legal QA with Re-Ranking

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**LegalAdapter** implements a **K-candidate generation + verifier re-ranking** framework for legal question answering. It improves accuracy and reduces hallucinations by generating multiple answers using diverse LLMs (OpenAI, xAI, Local models) and selecting the best one via a lightweight, trainable verifier.

> 🎓 **Course Project**: Final project for CS439 at Rutgers University.

## 🚀 Key Features

- **Multi-LLM Support**: OpenAI GPT-4, xAI Grok, local vLLM, and Ollama.
- **Verifier Re-Ranking**: Trains a BERT-size model to score and select the best answer.
- **Privacy Friendly**: Can run entirely on local hardware (open-source mode).
- **Evaluation Metrics**: Accuracy, F1, Verifier AUC, and Hallucination Rate.

---

## 🛠️ Prerequisites

- Python 3.8+
- **GPU**: Optional but recommended (4GB+ VRAM for training).
- **API Keys**: Optional (only if using proprietary models like OpenAI/xAI).

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/mikawaifu/ru-cs439-final-project.git
   cd ru-cs439-final-project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## ⚙️ Configuration

### 1. Environment Setup
Copy the template and configure your keys:
```bash
cp .env.example .env
```
Edit `.env` to add your API keys (leave empty if using local models only):
```ini
OPENAI_API_KEY=sk-...
XAI_API_KEY=...
OLLAMA_HOST=http://localhost:11434
```

### 2. Compute Device
To force CPU or specific GPU, edit `configs/base.yaml`:
```yaml
device: "cuda"  # or "cpu"
```

### 3. Model Selection
Enable/disable backends in `configs/models.yaml`:
```yaml
generators:
  openai_gpt4o:
    enabled: true   # Set true to use OpenAI
  xai_grok2:
    enabled: false
  ollama_mistral:
    enabled: false  # Set true for local Ollama
```

---

## ⚡ Quick Start

Run the full pipeline using synthetic test data in one command:

```bash
# 1. Generate synthetic test data
python -m src.legaladapter.utils.create_test_data

# 2. Run the entire pipeline (Generate -> Train -> Score -> Eval)
make test-quick
```

---

## 📖 Training Details (How it Works)

The system improves QA reliability through a two-stage process. Here is how the training and dataset loading works under the hood:

### 1. Dataset Loading
The system handles datasets defined in `configs/datasets.yaml` (e.g., COLIEE Task 4).
- **Raw Data**: Questions and ground truth answers are loaded via `DatasetLoader`.
- **Candidate Loading**: The training script (`train.py`) reads the generated candidates (from Step 1) located in `artifacts/candidates/`.
- **Labeling**: It automatically labels each candidate as **Positive (1)** or **Negative (0)** by comparing it to the ground truth:
  - **YES/NO Questions**: Checks if the candidate starts with the correct label.
  - **Open-Ended Questions**: Computes F1 score against ground truth (Positive if F1 ≥ 0.5).

### 2. Verifier Architecture
The verifier is a BERT-based binary classifier (default: `microsoft/deberta-v3-base`).
- **Input Format**: `[CLS] Question [SEP] Context [SEP] Candidate Answer [SEP]`
- **Training Objective**: Minimizes Binary Cross-Entropy (BCE) loss.
- **Optimization**: Uses AdamW optimizer with early stopping based on validation AUC.

### 3. Training Command
To train the verifier on your generated candidates:
```bash
python -m src.legaladapter.verify.train \
  --config configs/models.yaml \
  --datasets configs/datasets.yaml
```
This saves the best model to `artifacts/verifier/best_model.pt`.

---

## 📖 Step-by-Step Usage

For real datasets or custom workflows:

### 1. Generate Candidates
Generate K answers for each question using enabled LLMs:
```bash
python -m src.legaladapter.gen.generate --split test
```

### 2. Train Verifier
Train the scoring model on the training set (see details above):
```bash
python -m src.legaladapter.verify.train
```

### 3. Score & Re-Rank
Score candidates and select the best one:
```bash
python -m src.legaladapter.verify.infer --split test
python -m src.legaladapter.rank.rerank --split test
```

### 4. Evaluate
Calculate metrics and generate reports:
```bash
python -m src.legaladapter.eval.evaluate --split test
```

---

## 📂 Output & Artifacts

All results are saved in the `artifacts/` directory:

| File | Description |
|------|-------------|
| `artifacts/results/results_table_test.csv` | Summary metrics table |
| `artifacts/results/*_predictions.json` | Detailed predictions & scores |
| `artifacts/verifier/best_model.pt` | Trained verifier checkpoint |
| `artifacts/figures/*.png` | Visualization plots |

---

## 📝 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
