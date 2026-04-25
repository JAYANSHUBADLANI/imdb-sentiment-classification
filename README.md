# IMDB Sentiment Classification

A progressive comparison of three NLP approaches on the [IMDB 50K Movie Reviews](https://huggingface.co/datasets/imdb) dataset — from a classical ML baseline through deep learning and up to a fine-tuned transformer.

---

## Project Structure

```
imdb-sentiment-classification/
├── baseline.ipynb          # TF-IDF + Logistic Regression
├── lstm.ipynb              # Bidirectional LSTM + GloVe 100d
├── bert.ipynb              # DistilBERT fine-tuning (HuggingFace + PyTorch)
├── requirements.txt
├── results/
│   ├── baseline_metrics.json
│   ├── lstm_metrics.json
│   ├── bert_metrics.json
│   ├── baseline_confusion_matrix.png
│   ├── baseline_top_features.png
│   ├── lstm_training_curves.png
│   ├── lstm_confusion_matrix.png
│   ├── bert_training_curves.png
│   └── bert_confusion_matrix.png
└── README.md
```

---

## Model Comparison

| Model | Accuracy | F1 Score | Latency / Sample | Notes |
|---|---|---|---|---|
| TF-IDF + Logistic Regression | ~0.8970 | ~0.8972 | ~0.01 ms | Strong baseline; interpretable |
| Bidirectional LSTM + GloVe | ~0.8850 | ~0.8848 | ~2.5 ms | Sequential context; pre-trained embeddings |
| DistilBERT (fine-tuned) | ~0.9320 | ~0.9321 | ~12 ms | Best accuracy; highest compute cost |

> Exact numbers are written to `results/` after each notebook runs.

---

## Approach Summary

### 1. `baseline.ipynb` — TF-IDF + Logistic Regression
- Regex-based text cleaning (HTML stripping, lowercasing, punctuation removal)
- `TfidfVectorizer` with unigrams + bigrams, 50k vocabulary, sublinear TF scaling
- `LogisticRegression` (L2, lbfgs solver)
- Outputs: accuracy, F1, confusion matrix, top predictive features

### 2. `lstm.ipynb` — Bidirectional LSTM + GloVe
- Same preprocessing pipeline
- Keras `Tokenizer` → integer sequences → padding to length 256
- GloVe 6B 100d embeddings (auto-downloaded) loaded into a frozen `Embedding` layer
- Two stacked `Bidirectional(LSTM)` layers + Dense head
- `EarlyStopping` + `ReduceLROnPlateau` callbacks
- Outputs: training curves (accuracy + loss), confusion matrix

### 3. `bert.ipynb` — DistilBERT Fine-tuning
- `DistilBertTokenizerFast` with truncation/padding to 256 tokens
- `DistilBertForSequenceClassification` (`distilbert-base-uncased`, 66M params)
- HuggingFace `Trainer` with warmup, weight decay, early stopping
- Mixed precision (fp16) on GPU automatically enabled
- Per-sample inference latency benchmark (100-run average)
- Demo inference function included

---

## Setup

### Prerequisites
- Python 3.10+
- (Optional but recommended) CUDA-capable GPU for BERT training

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run notebooks in order

```bash
jupyter notebook baseline.ipynb
jupyter notebook lstm.ipynb
jupyter notebook bert.ipynb
```

Each notebook saves its metrics to `results/` as a JSON file and saves plots as PNGs.

---

## Key Design Decisions

- **GloVe download** is handled automatically in `lstm.ipynb` if the file is not found locally.
- **DistilBERT** is used over full BERT for faster training (~40% fewer parameters) with minimal accuracy trade-off.
- **`report_to='none'`** in TrainingArguments keeps BERT training output clean without requiring wandb/tensorboard.
- All random seeds are fixed (`RANDOM_SEED = 42`) for reproducibility.

---

## Author

**Jayanshu Badlani**  
Data Science Enthusiast | Justdial  
[GitHub](https://github.com/JAYANSHUBADLANI)
