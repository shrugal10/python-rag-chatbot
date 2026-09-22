import os
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI

DB_DIR = "chroma_db"
COLLECTION = "python_docs"
EMBED_MODEL = "all-MiniLM-L6-v2"
GEN_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

_model = SentenceTransformer(EMBED_MODEL)
_client = chromadb.PersistentClient(path=DB_DIR)

def get_collection():
    return _client.get_collection(COLLECTION)

def retrieve(question, k=5):
    collection = get_collection()
    embedding = _model.encode([question], normalize_embeddings=True).tolist()[0]
    result = collection.query(
        query_embeddings=[embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    for doc, meta, distance in zip(
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0]
    ):
        chunks.append({
            "text": doc,
            "file": meta["file"],
            "page": meta["page"],
            "distance": distance
        })
    return chunks

def answer_question(question):
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing.")

    chunks = retrieve(question, k=5)

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[SOURCE {i}] {chunk['file']} — Page {chunk['page']}\n{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a domain-specific document assistant.

Answer the user's question ONLY from the supplied document context.
If the context does not contain enough information to answer, respond exactly:
"I can't answer that from the provided document corpus."

Do not use outside knowledge.
Do not guess.
Keep the answer concise and cite supporting sources using [SOURCE N].

Question:
{question}

Document context:
{context}
"""

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.responses.create(
        model=GEN_MODEL,
        input=prompt
    )

    answer = response.output_text.strip()

    sources = [
        {
            "file": c["file"],
            "page": c["page"],
            "preview": c["text"][:240] + ("..." if len(c["text"]) > 240 else "")
        }
        for c in chunks
    ]

    return {"answer": answer, "sources": sources}
