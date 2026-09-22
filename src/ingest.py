import re
import uuid
from pathlib import Path

import chromadb
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer


HTML_DIR = Path("data/_python_docs")
DB_DIR = "chroma_db"
COLLECTION = "python_docs"
EMBED_MODEL = "all-MiniLM-L6-v2"


def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text, target_words=350, overlap_words=60):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(len(words), start + target_words)

        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        if end == len(words):
            break

        start = max(0, end - overlap_words)

    return chunks


def html_files():
    if not HTML_DIR.exists():
        return []

    return list(HTML_DIR.rglob("*.html"))


def ingest():
    print("Loading embedding model...")

    model = SentenceTransformer(EMBED_MODEL)

    print("Connecting to ChromaDB...")

    client = chromadb.PersistentClient(path=DB_DIR)

    # Delete the old collection so every ingestion starts clean.
    try:
        client.delete_collection(COLLECTION)
        print("Old ChromaDB collection deleted.")
    except Exception:
        pass

    collection = client.get_or_create_collection(COLLECTION)

    files = html_files()

    if not files:
        raise RuntimeError(
            "Python documentation not found. "
            "Run: python scripts/download_python_docs.py"
        )

    print(f"Found {len(files)} HTML documentation files.")

    documents = []
    ids = []
    metadatas = []

    print("Creating chunks...")

    for path in files:

        try:
            html = path.read_text(
                encoding="utf-8",
                errors="ignore"
            )

            soup = BeautifulSoup(
                html,
                "html.parser"
            )

            # Remove unnecessary HTML elements.
            for tag in soup(["script", "style", "nav"]):
                tag.decompose()

            title = (
                soup.title.get_text(" ", strip=True)
                if soup.title
                else path.stem
            )

            text = clean_text(
                soup.get_text(" ")
            )

            chunks = chunk_text(text)

            for i, chunk in enumerate(chunks):

                documents.append(chunk)

                # Guaranteed unique Chroma ID.
                chunk_id = f"{uuid.uuid4()}"

                ids.append(chunk_id)

                metadatas.append(
                    {
                        "file": path.name,
                        "page": 1,
                        "title": title,
                    }
                )

        except Exception as error:
            print(
                f"Skipping {path.name}: {error}"
            )

    print(f"Created {len(documents)} chunks.")

    if not documents:
        raise RuntimeError(
            "No text chunks were created from the documentation."
        )

    print("Creating embeddings...")

    embeddings = model.encode(
        documents,
        normalize_embeddings=True,
        show_progress_bar=True
    ).tolist()

    # Chroma has a maximum batch size.
    # 4000 keeps us safely below the limit.
    batch_size = 4000

    print("Adding chunks to ChromaDB...")

    for start in range(
        0,
        len(documents),
        batch_size
    ):

        end = min(
            start + batch_size,
            len(documents)
        )

        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )

        print(
            f"Added chunks "
            f"{start + 1}-{end} "
            f"/ {len(documents)}"
        )

    print()
    print("=" * 60)
    print("INGESTION COMPLETE!")
    print("=" * 60)
    print(
        f"Total documents: {len(files)}"
    )
    print(
        f"Total chunks: {len(documents)}"
    )
    print(
        f"Chroma collection: {COLLECTION}"
    )
    print("=" * 60)


if __name__ == "__main__":
    ingest()