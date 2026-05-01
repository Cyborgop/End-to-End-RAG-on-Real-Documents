import numpy as np


def cosine_similarity(vec1, vec2):
    a, b = np.array(vec1, dtype="float32"), np.array(vec2, dtype="float32")
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def retrieval_precision(retrieved_chunks, relevant_texts):
    """Fraction of retrieved chunks that overlap with at least one relevant text."""
    if not retrieved_chunks:
        return 0.0
    hits = sum(
        any(rel.lower() in chunk.lower() or chunk.lower() in rel.lower()
            for rel in relevant_texts)
        for chunk in retrieved_chunks
    )
    return hits / len(retrieved_chunks)


def answer_faithfulness(answer, context_chunks, embed_fn):
    """Semantic similarity between the generated answer and the retrieved context."""
    context = " ".join(context_chunks)
    answer_emb = embed_fn([answer])[0]
    context_emb = embed_fn([context])[0]
    return cosine_similarity(answer_emb, context_emb)


def evaluate(answer, context_chunks, embed_fn, relevant_texts=None):
    """Return a dict of all available metrics for a single RAG response."""
    results = {
        "faithfulness": answer_faithfulness(answer, context_chunks, embed_fn),
    }
    if relevant_texts:
        results["retrieval_precision"] = retrieval_precision(context_chunks, relevant_texts)
    return results
