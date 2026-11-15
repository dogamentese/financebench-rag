"""Embedding search over the chunks, with Qdrant as the vector store."""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


class EmbeddingSearch:
    """Meaning-based search: chunks and questions become vectors, and the closest chunks win."""

    def __init__(self, chunks: list[dict], model, batch_size: int = 64):
        self.chunks = chunks
        self.model = model
        vectors = model.encode([c["text"] for c in chunks], batch_size=batch_size, show_progress_bar=True)
        self.client = QdrantClient(":memory:")
        self.client.create_collection("chunks", vectors_config=VectorParams(size=vectors.shape[1], distance=Distance.COSINE))
        for start in range(0, len(chunks), 1000):
            points = [PointStruct(id=i, vector=vectors[i].tolist()) for i in range(start, min(start + 1000, len(chunks)))]
            self.client.upsert("chunks", points)

    def search(self, question: str, k: int = 10, n_chunks: int = 100) -> list[tuple[str, int]]:
        """Return the k best pages for a question. A page is ranked by its best scoring chunk."""
        vector = self.model.encode(question, prompt_name="query")
        hits = self.client.query_points("chunks", query=vector.tolist(), limit=n_chunks).points
        pages = []
        for hit in hits:
            page = (self.chunks[hit.id]["doc_name"], self.chunks[hit.id]["page_num"])
            if page not in pages:
                pages.append(page)
        return pages[:k]
