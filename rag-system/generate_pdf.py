from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

OUTPUT = "RAG_Project_Summary.pdf"

# ── Styles ──────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

def style(name, **kw):
    return ParagraphStyle(name, parent=styles["Normal"], **kw)

TITLE    = style("Title",    fontSize=26, textColor=colors.HexColor("#1a1a2e"),
                 alignment=TA_CENTER, spaceAfter=6, fontName="Helvetica-Bold")
SUBTITLE = style("Subtitle", fontSize=13, textColor=colors.HexColor("#4a4e69"),
                 alignment=TA_CENTER, spaceAfter=20, fontName="Helvetica")
H1       = style("H1",       fontSize=16, textColor=colors.HexColor("#1a1a2e"),
                 spaceBefore=18, spaceAfter=6, fontName="Helvetica-Bold")
H2       = style("H2",       fontSize=12, textColor=colors.HexColor("#22223b"),
                 spaceBefore=12, spaceAfter=4, fontName="Helvetica-Bold")
Q_STYLE  = style("Q",        fontSize=11, textColor=colors.HexColor("#c9184a"),
                 spaceBefore=10, spaceAfter=2, fontName="Helvetica-Bold",
                 leftIndent=10)
A_STYLE  = style("A",        fontSize=10, textColor=colors.HexColor("#22223b"),
                 spaceAfter=6,  fontName="Helvetica",
                 leftIndent=10, alignment=TA_JUSTIFY, leading=14)
CODE     = style("Code",     fontSize=9,  textColor=colors.HexColor("#006400"),
                 fontName="Courier", leftIndent=20, spaceAfter=6,
                 backColor=colors.HexColor("#f0f4f0"), leading=13)
BODY     = style("Body",     fontSize=10, textColor=colors.HexColor("#22223b"),
                 spaceAfter=5,  fontName="Helvetica", leading=14,
                 alignment=TA_JUSTIFY)

HR = HRFlowable(width="100%", thickness=1, color=colors.HexColor("#c9184a"), spaceAfter=8)

def h(text, s=H1): return Paragraph(text, s)
def p(text, s=BODY): return Paragraph(text, s)
def q(text): return Paragraph(f"Q: {text}", Q_STYLE)
def a(text): return Paragraph(f"A: {text}", A_STYLE)
def code(text): return Paragraph(text.replace(" ", "&nbsp;").replace("\n", "<br/>"), CODE)
def sp(n=8): return Spacer(1, n)

# ── Architecture table ───────────────────────────────────────────────────────
ARCH_DATA = [
    ["Stage", "Module", "Key Function", "Library"],
    ["1 · Ingestion",  "ingestion/pdf_loader.py",  "load_pdf(file_path)",                 "PyPDF2"],
    ["",               "ingestion/chunking.py",    "chunk_text(text, size=500, overlap=100)", "—"],
    ["2 · Embedding",  "embeddings/embedder.py",   "get_embeddings(chunks)",              "sentence-transformers"],
    ["3 · Indexing",   "retrieval/faiss_index.py", "create_faiss_index(embeddings)",      "FAISS"],
    ["4 · Retrieval",  "retrieval/retrieval.py",   "retrieve(query, model, index, chunks, k=3)", "FAISS"],
    ["5 · Generation", "generation/llm_pipeline.py","generate_answer(query, context_chunks)","OpenAI API"],
    ["6 · Evaluation", "evaluation/metrics.py",    "evaluate(answer, context, embed_fn)", "NumPy"],
]
ARCH_STYLE = TableStyle([
    ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
    ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
    ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE",    (0, 0), (-1, 0), 9),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1),
     [colors.HexColor("#f8f9fa"), colors.HexColor("#e9ecef")]),
    ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
    ("FONTSIZE",    (0, 1), (-1, -1), 8),
    ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#ced4da")),
    ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING",  (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
    ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ("SPAN",        (0, 1), (0, 2)),
])

# ── Q&A data ─────────────────────────────────────────────────────────────────
QA = [
    # ── Conceptual ──
    ("What is Retrieval-Augmented Generation (RAG)?",
     "RAG is an AI framework that combines a retrieval system with a generative LLM. "
     "Instead of relying solely on the model's parametric knowledge, RAG first retrieves "
     "relevant passages from an external document store and injects them into the prompt, "
     "allowing the model to ground its answer in up-to-date, domain-specific content."),

    ("Why use RAG instead of fine-tuning?",
     "Fine-tuning embeds knowledge into model weights — expensive, slow to update, and prone "
     "to catastrophic forgetting. RAG externalises knowledge into a retrieval index that can "
     "be updated cheaply without retraining the model. RAG also provides citations and is "
     "more transparent about its information source."),

    ("What problem does chunking solve, and what are the tradeoffs?",
     "LLMs have a fixed context window, so we cannot feed an entire document at once. Chunking "
     "splits the document into pieces small enough to embed and compare. "
     "Smaller chunks = more precise retrieval but lose surrounding context. "
     "Larger chunks = richer context per hit but noisier similarity scores. "
     "The overlap parameter (100 chars here) prevents information loss at chunk boundaries."),

    ("Which embedding model is used and why?",
     "'all-MiniLM-L6-v2' from sentence-transformers. It is a lightweight (22 M params) "
     "transformer that produces 384-dimensional unit-normalised embeddings. It balances speed "
     "and quality well for English semantic search tasks and is freely available offline — no "
     "API call needed for embedding."),

    ("Why is FAISS IndexFlatL2 equivalent to cosine similarity here?",
     "Cosine similarity = dot product of unit vectors. The MiniLM model returns L2-normalised "
     "embeddings (||v|| = 1). For unit vectors, ||a - b||² = 2 - 2·(a·b), which is a monotonic "
     "transformation of cosine similarity. Ranking by L2 distance therefore gives the same "
     "ordering as ranking by cosine similarity — no separate normalisation step needed."),

    ("What does IndexFlatL2 do internally?",
     "It performs an exact brute-force L2 distance scan over all stored vectors. It makes no "
     "approximation, so recall = 100%, but query time scales linearly O(n·d) with the number "
     "of vectors n and dimension d. For large corpora, approximate indexes like IVFFlat or HNSW "
     "are preferred."),

    ("How does retrieval work at query time?",
     "1. The user query is encoded with the same SentenceTransformer model into a 384-d vector. "
     "2. FAISS performs an L2 nearest-neighbour search and returns the k=3 closest chunk vectors. "
     "3. The corresponding text chunks are assembled as context and passed to the LLM alongside "
     "the original question."),

    ("What prompt strategy is used for generation?",
     "A two-message chat prompt: a system message instructing the model to answer only from the "
     "provided context and to admit uncertainty when context is insufficient, and a user message "
     "containing the retrieved context blocks followed by the question. Temperature is set to 0 "
     "for deterministic, factual answers."),

    ("Explain the faithfulness metric.",
     "Faithfulness measures how semantically consistent the generated answer is with the retrieved "
     "context. Both the answer and the concatenated context are embedded with the same "
     "SentenceTransformer model, then their cosine similarity is computed. A score near 1 means "
     "the answer closely reflects the context; a low score may indicate hallucination."),

    ("What is retrieval_precision in the evaluation module?",
     "It is the fraction of retrieved chunks that contain (or are contained in) at least one "
     "known-relevant text snippet. It requires a ground-truth list of relevant passages. "
     "Precision = |relevant ∩ retrieved| / |retrieved|. This is useful for offline evaluation "
     "when labelled data is available."),

    ("What are the limitations of this RAG implementation?",
     "1. Character-level chunking can split words and sentences. "
     "2. IndexFlatL2 does not scale past ~100 k vectors without slowdown. "
     "3. No persistent index — it is rebuilt on every run. "
     "4. Single-document only; multi-document retrieval is not handled. "
     "5. Evaluation metrics are basic; no BLEU/ROUGE or human eval. "
     "6. No re-ranking step after initial retrieval."),

    ("How would you scale this to production?",
     "Replace IndexFlatL2 with an IVFFlat or HNSW index and persist it to disk. "
     "Add a metadata store (SQLite/PostgreSQL) to map chunk IDs back to source documents and page numbers. "
     "Introduce a re-ranker (cross-encoder) between retrieval and generation to improve precision. "
     "Use a vector database (Pinecone, Weaviate, pgvector) for managed, scalable storage. "
     "Cache embeddings so re-indexing only processes changed documents."),

    ("How would you improve chunking quality?",
     "Switch from character-level windows to sentence-aware splitting (e.g., using spaCy or NLTK "
     "sentence tokenisation), or semantic chunking that groups sentences by topic similarity. "
     "Hierarchical chunking (large parent + small child chunks) can also be used: retrieve small "
     "chunks for precision, but send the parent chunk to the LLM for context richness."),

    ("What is the role of the overlap parameter?",
     "Overlap ensures that information near chunk boundaries is not lost. With chunk_size=500 and "
     "overlap=100, consecutive chunks share 100 characters. This means a sentence split across two "
     "chunks will appear in full in at least one of them, improving recall for boundary-spanning queries."),

    ("How is the SentenceTransformer model loaded, and why does it matter?",
     "It is loaded as a module-level global in embedder.py — once on first import. This is "
     "efficient for a single-process pipeline because the model (≈80 MB) is loaded exactly once. "
     "In a multi-worker web server, each worker process would load its own copy; for that case, "
     "a shared model server (Triton, TorchServe) or lazy singleton with a lock would be better."),

    ("What would you add to make this system production-ready?",
     "Async query handling, streaming LLM responses, document-level access control, "
     "query logging and feedback collection for continuous improvement, "
     "a REST API layer (FastAPI), a simple frontend (Streamlit or React), "
     "automated re-indexing on document upload, and integration tests with a fixed document "
     "and known ground-truth answers to catch regressions."),
]

# ── Build document ────────────────────────────────────────────────────────────
def build():
    doc = SimpleDocTemplate(
        OUTPUT, pagesize=A4,
        topMargin=2*cm, bottomMargin=2*cm,
        leftMargin=2.2*cm, rightMargin=2.2*cm,
    )
    story = []

    # Cover
    story += [
        sp(40),
        Paragraph("End-to-End RAG on Real Documents", TITLE),
        Paragraph("Project Summary &amp; Interview Q&amp;A", SUBTITLE),
        HR,
        sp(6),
        p("A complete Retrieval-Augmented Generation pipeline: PDF ingestion → "
          "chunking → embedding → FAISS indexing → semantic retrieval → "
          "LLM generation → faithfulness evaluation.", BODY),
        sp(4),
        p("<b>Tech stack:</b> Python · PyPDF2 · sentence-transformers · FAISS · OpenAI API · NumPy", BODY),
        PageBreak(),
    ]

    # Architecture
    story += [
        h("Pipeline Architecture"),
        HR,
        p("The system processes a PDF through six sequential stages. Each stage is "
          "isolated in its own module so components can be swapped independently."),
        sp(8),
        Table(ARCH_DATA,
              colWidths=[3.2*cm, 4.8*cm, 5.5*cm, 3.5*cm],
              style=ARCH_STYLE),
        sp(14),
    ]

    # Data flow
    story += [
        h("Data Flow"),
        HR,
        p("1. <b>load_pdf</b> extracts raw text from every page via PyPDF2."),
        p("2. <b>chunk_text</b> splits the text into 500-char windows with 100-char overlap."),
        p("3. <b>get_embeddings</b> encodes each chunk into a 384-d float32 vector."),
        p("4. <b>create_faiss_index</b> adds all vectors into an IndexFlatL2 index."),
        p("5. At query time, <b>retrieve</b> encodes the question, runs a k-NN search (k=3), "
          "and returns the nearest chunks as context."),
        p("6. <b>generate_answer</b> sends the context + question to GPT-3.5-Turbo (temp=0)."),
        p("7. <b>evaluate</b> scores the answer for faithfulness via cosine similarity."),
        sp(8),
        h("Running the App", H2),
        code("pip install -r requirements.txt\n"
             "export OPENAI_API_KEY=sk-...\n"
             "python app.py path/to/document.pdf"),
        PageBreak(),
    ]

    # Q&A
    story += [h("Interview Questions &amp; Answers"), HR]
    for i, (question, answer) in enumerate(QA, 1):
        story += [q(f"[{i}]  {question}"), a(answer), sp(4)]

    doc.build(story)
    print(f"PDF saved: {OUTPUT}")

if __name__ == "__main__":
    build()
