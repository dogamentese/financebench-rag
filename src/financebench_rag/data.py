"""Loading the FinanceBench questions, the dev/test split and the page text of the corpus."""

import json
from pathlib import Path

import pymupdf

REPO = Path(__file__).resolve().parents[2]
FB = REPO / "data" / "financebench"
PAGES_PATH = REPO / "data" / "processed" / "pages.jsonl"


def load_questions() -> list[dict]:
    """The 150 questions of the FinanceBench open-source subset."""
    return [json.loads(line) for line in open(FB / "financebench_open_source.jsonl")]


def load_split() -> dict:
    """The frozen dev/test split, with the question ids under "dev" and "test"."""
    return json.loads((REPO / "data" / "split.json").read_text())


def load_doc_info() -> dict[str, dict]:
    """Company, document type and fiscal year of every document, keyed by doc_name."""
    docs = (json.loads(line) for line in open(FB / "financebench_document_information.jsonl"))
    return {d["doc_name"]: d for d in docs}


def extract_pages() -> None:
    """Extract the text of every page of the corpus PDFs into data/processed/pages.jsonl."""
    PAGES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PAGES_PATH, "w") as f:
        for pdf in sorted((FB / "pdfs").glob("*.pdf")):
            for page_num, page in enumerate(pymupdf.open(pdf)):
                row = {"doc_name": pdf.stem, "page_num": page_num, "text": page.get_text()}
                f.write(json.dumps(row) + "\n")


def load_pages() -> list[dict]:
    """One dict per page of the corpus, with doc_name, page_num (0-based) and text."""
    return [json.loads(line) for line in open(PAGES_PATH)]


def gold_pages(question: dict) -> set[tuple[str, int]]:
    """The evidence pages of a question as (doc_name, page_num) pairs."""
    return {(question["doc_name"], ev["evidence_page_num"]) for ev in question["evidence"]}
