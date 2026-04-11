def retrieve(query, model, index, chunks, k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)
    results = [chunks[i] for i in indices[0]]

    return results