"""Retrieval metrics against the evidence pages."""

from financebench_rag.data import gold_pages


def recall_at_k(ranked: list[tuple[str, int]], gold: set[tuple[str, int]], k: int) -> float:
    """1.0 if any evidence page is among the first k results, otherwise 0.0."""
    return float(any(page in gold for page in ranked[:k]))


def reciprocal_rank(ranked: list[tuple[str, int]], gold: set[tuple[str, int]]) -> float:
    """1 / rank of the first evidence page in the results, or 0.0 if none was found."""
    for rank, page in enumerate(ranked, start=1):
        if page in gold:
            return 1 / rank
    return 0.0


def evaluate_retrieval(results: dict[str, list[tuple[str, int]]], questions: list[dict]) -> dict[str, float]:
    """Average the retrieval metrics over the questions that have an entry in `results`."""
    rows = []
    for q in questions:
        if q["financebench_id"] not in results:
            continue
        ranked = results[q["financebench_id"]]
        gold = gold_pages(q)
        rows.append({
            "recall@1": recall_at_k(ranked, gold, 1),
            "recall@5": recall_at_k(ranked, gold, 5),
            "mrr@10": reciprocal_rank(ranked[:10], gold),
            "doc_recall@5": float(any(doc == q["doc_name"] for doc, _ in ranked[:5])),
        })
    return {metric: round(sum(r[metric] for r in rows) / len(rows), 3) for metric in rows[0]}
