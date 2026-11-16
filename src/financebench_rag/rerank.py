"""Reranking candidate pages with a cross-encoder."""

from sentence_transformers import CrossEncoder


class Reranker:
    """Scores question and chunk together, then re-sorts the candidate pages."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model = CrossEncoder(model_name, max_length=1024)

    def rerank(self, question: str, candidates: list[tuple[str, int]], chunks_by_page: dict, k: int = 10) -> list[tuple[str, int]]:
        """Re-sort candidate pages. A page is scored by its best chunk."""
        pairs, owners = [], []
        for page in candidates:
            for chunk in chunks_by_page[page]:
                pairs.append((question, chunk["text"]))
                owners.append(page)
        scores = self.model.predict(pairs, batch_size=16, show_progress_bar=False)
        best = {}
        for page, score in zip(owners, scores):
            best[page] = max(best.get(page, float(score)), float(score))
        return sorted(best, key=best.get, reverse=True)[:k]
