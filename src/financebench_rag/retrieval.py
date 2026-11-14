"""Chunking the page text and searching it with BM25."""

import bm25s


def chunk_text(text: str, size: int = 250, overlap: int = 50) -> list[str]:
    """Cut text into chunks of `size` words. Neighbouring chunks share `overlap` words."""
    words = text.split()
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, max(len(words) - overlap, 1), step)]


def build_chunks(pages: list[dict], doc_info: dict, size: int = 250, overlap: int = 50, prefix: bool = True) -> list[dict]:
    """Chunk every page of the corpus.

    With prefix=True each chunk starts with the company, document type and fiscal year
    of its report, so these stay attached to every piece of the page.
    """
    chunks = []
    for page in pages:
        info = doc_info[page["doc_name"]]
        head = f"{info['company']} {info['doc_type']} FY{info['doc_period']} {info['doc_period']}\n" if prefix else ""
        for text in chunk_text(page["text"], size, overlap):
            if text:
                chunks.append({"doc_name": page["doc_name"], "page_num": page["page_num"], "text": head + text})
    return chunks


class BM25Search:
    """BM25 keyword search over chunks, returning pages."""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.index = bm25s.BM25()
        self.index.index(bm25s.tokenize([c["text"] for c in chunks], stopwords="en", show_progress=False), show_progress=False)

    def search(self, question: str, k: int = 10, n_chunks: int = 100) -> list[tuple[str, int]]:
        """Return the k best pages for a question. A page is ranked by its best scoring chunk."""
        tokens = bm25s.tokenize(question, stopwords="en", show_progress=False)
        chunk_ids, _ = self.index.retrieve(tokens, k=n_chunks, show_progress=False)
        pages = []
        for i in chunk_ids[0]:
            page = (self.chunks[i]["doc_name"], self.chunks[i]["page_num"])
            if page not in pages:
                pages.append(page)
        return pages[:k]
