"""Download the FinanceBench open-source subset into data/financebench/.

Run once from the repo root:

    python scripts/download_data.py

Downloads the question file, the document list, and the 84 PDFs that the
150 questions reference (about 160 MB).
Source: https://github.com/patronus-ai/financebench
"""

import json
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/patronus-ai/financebench/main"
FB = Path(__file__).resolve().parents[1] / "data" / "financebench"
PDF_DIR = FB / "pdfs"
PDF_DIR.mkdir(parents=True, exist_ok=True)

urllib.request.urlretrieve(f"{BASE}/data/financebench_open_source.jsonl", FB / "financebench_open_source.jsonl")
urllib.request.urlretrieve(f"{BASE}/data/financebench_document_information.jsonl", FB / "financebench_document_information.jsonl")

questions = [json.loads(line) for line in open(FB / "financebench_open_source.jsonl")]
doc_names = sorted({q["doc_name"] for q in questions})

for i, name in enumerate(doc_names, 1):
    print(f"[{i}/{len(doc_names)}] {name}")
    urllib.request.urlretrieve(f"{BASE}/pdfs/{name}.pdf", PDF_DIR / f"{name}.pdf")

print(f"done: {len(questions)} questions, {len(doc_names)} PDFs")