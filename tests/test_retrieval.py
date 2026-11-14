from financebench_rag.evaluate import recall_at_k, reciprocal_rank
from financebench_rag.retrieval import chunk_text


def test_chunks_have_the_requested_size_and_overlap():
    words = [str(i) for i in range(600)]
    chunks = chunk_text(" ".join(words), size=250, overlap=50)
    assert [len(c.split()) for c in chunks] == [250, 250, 200]
    assert chunks[0].split()[-50:] == chunks[1].split()[:50]


def test_every_word_lands_in_a_chunk():
    words = [str(i) for i in range(1234)]
    chunks = chunk_text(" ".join(words), size=250, overlap=50)
    assert {w for c in chunks for w in c.split()} == set(words)


def test_short_text_is_one_chunk():
    assert chunk_text("net sales were up", size=250, overlap=50) == ["net sales were up"]


def test_recall_counts_any_evidence_page():
    ranked = [("A", 1), ("A", 2), ("B", 7)]
    assert recall_at_k(ranked, {("B", 7), ("C", 3)}, k=3) == 1.0
    assert recall_at_k(ranked, {("B", 7)}, k=2) == 0.0


def test_reciprocal_rank():
    ranked = [("A", 1), ("A", 2), ("B", 7)]
    assert reciprocal_rank(ranked, {("A", 2)}) == 0.5
    assert reciprocal_rank(ranked, {("Z", 0)}) == 0.0
