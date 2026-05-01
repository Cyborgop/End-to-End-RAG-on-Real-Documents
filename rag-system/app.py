import os
import sys

from ingestion.pdf_loader import load_pdf
from ingestion.chunking import chunk_text
from embeddings.embedder import model as embed_model, get_embeddings
from retrieval.faiss_index import create_faiss_index
from retrieval.retrieval import retrieve
from generation.llm_pipeline import generate_answer
from evaluation.metrics import evaluate


def build_index(pdf_path):
    print(f"Loading: {pdf_path}")
    text = load_pdf(pdf_path)
    chunks = chunk_text(text)
    print(f"Chunked into {len(chunks)} segments. Embedding...")
    embeddings = get_embeddings(chunks)
    index = create_faiss_index(embeddings)
    print("Index ready.\n")
    return index, chunks


def answer_query(question, index, chunks, k=3, show_eval=False):
    context = retrieve(question, embed_model, index, chunks, k)
    answer = generate_answer(question, context)

    print(f"\nAnswer:\n{answer}\n")

    if show_eval:
        scores = evaluate(answer, context, get_embeddings)
        print("Evaluation:")
        for metric, value in scores.items():
            print(f"  {metric}: {value:.3f}")
        print()

    return answer, context


def main():
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = input("PDF path: ").strip()

    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    index, chunks = build_index(pdf_path)

    while True:
        try:
            question = input("Question (q to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in ("q", "quit", "exit"):
            break
        if not question:
            continue
        answer_query(question, index, chunks, show_eval=True)


if __name__ == "__main__":
    main()
