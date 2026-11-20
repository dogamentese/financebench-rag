# Text or Image? Reading Financial Reports with Language Models

Retrieval-augmented question answering over company reports, on the open-source subset of FinanceBench (150 questions, 84 reports). Part A compares retrieval methods for finding the evidence page. Part B tests whether a vision language model answers more accurately from the extracted text of a page or from its image.

Course project for "Essentials in Text and Speech Processing", University of Zurich, fall 2026, by Doğa Menteşe and Tuna Özdemir.

## Main results

On the 120 test questions:

- Dense embeddings find an evidence page in the top 5 for 67.5% of the questions, against 15.8% for BM25. Fusion and reranking do not improve on embeddings.
- With the evidence pages, Qwen3.5-4B answers 78.3% of the questions correctly from text and 80.0% from images. The difference is not statistically reliable.
- With retrieved pages, accuracy drops by about 23 percentage points in both formats, so retrieval limits performance more than the input format.

## Models

No model was trained. All models are used as published.

| Role | Model |
|---|---|
| Keyword search | BM25 via [`bm25s`](https://github.com/xhluca/bm25s) |
| Dense embeddings | [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) |
| Reranker | [bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) |
| Reader | [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), thinking off, greedy decoding, up to 1024 new tokens, pages rendered at 150 dpi |

## Repository structure

```
data/split.json            fixed 30/120 dev/test split
notebooks/                 experiments, numbered in the order they were run
results/                   retrieval rankings, model answers and grading labels
scripts/download_data.py   downloads the questions and reports
src/financebench_rag/      code shared by the notebooks
tests/                     unit tests for retrieval and grading
```

| Module | Contents |
|---|---|
| `data.py` | Loading questions, the split and page text |
| `retrieval.py` | Chunking, BM25 and reciprocal rank fusion |
| `embedding.py` | Dense search with a Qdrant vector database |
| `rerank.py` | Cross-encoder reranking |
| `evaluate.py` | Recall@k and MRR |
| `reader.py` | The reader, with pages given as text or as images |
| `grading.py` | Automatic answer labels (final answer, refusal, 3% numeric match) |

## Setup

Requires Python 3.10 or newer.

```bash
git clone https://github.com/dogamentese/financebench-rag.git
cd financebench-rag
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
python scripts/download_data.py
```

The download script fetches the 150 questions and the 84 reports they refer to (about 160 MB) from the official FinanceBench repository into `data/financebench/`.

## Reproducing the results

All settings were chosen on the 30 dev questions. The test questions were only used in notebooks 06 and 08.

Notebooks marked GPU were run on Google Colab with a Tesla T4. They clone the repository and install their own dependencies (transformers, sentence-transformers, qdrant-client) in the first cell. `requirements.txt` covers the CPU notebooks and the tests.

| Notebook | Step | Runs on | Writes to `results/` |
|---|---|---|---|
| `01_data_exploration` | Checks the page labels and fixes the dev/test split | CPU | `data/split.json` |
| `02_reader_feasability` | Tests the reader on three dev questions | GPU | nothing |
| `03_search_bm25` | BM25, chunk size and prefix | CPU | `dev_retrieval_bm25.json` |
| `04_search_embeddings` | Dense embeddings | GPU | `dev_retrieval_embeddings.json` |
| `05_search_hybrid_rerank` | Fusion and reranking | GPU | `dev_retrieval_hybrid.json` |
| `06_search_test_results` | All retrieval methods on the test questions | GPU | `test_retrieval_*.json` |
| `07_reader_text_image_dev` | All five reader settings on the dev questions | GPU | `dev_answers_*.json` |
| `08_reader_text_image_test_runs/` | One notebook per reader setting on the test questions | GPU | `test_answers_*.json` |
| `09_reader_grading` | Code labels and hand-checked labels | CPU | `test_labels.csv` |
| `10_reader_text_vs_image_results` | Accuracy, confidence intervals and comparisons | CPU | nothing |

Files named `hybrid` hold the fused + rerank rankings, which are passed to the reader.

All outputs are committed in `results/`, so notebooks 09 and 10 can be rerun without a GPU. The reader uses greedy decoding, so rerunning a notebook in 08 gives the same answers.

## Tests

```bash
pytest
```

## Data

The questions and reports come from FinanceBench (Islam et al., 2023, [arXiv:2311.11944](https://arxiv.org/abs/2311.11944)), available at [patronus-ai/financebench](https://github.com/patronus-ai/financebench).