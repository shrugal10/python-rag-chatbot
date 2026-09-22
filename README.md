# Domain-Specific RAG Chatbot

A small Retrieval-Augmented Generation (RAG) chatbot that answers Python questions from the official Python 3.14 documentation and refuses questions that cannot be supported by the corpus.

## Architecture

PDF documents
→ text extraction with PyMuPDF
→ overlapping chunks
→ Sentence-Transformer embeddings
→ ChromaDB vector store
→ top-5 semantic retrieval
→ OpenAI Responses API
→ grounded answer + source pages

## Corpus

This project uses the official Python 3.14.7 documentation as its domain corpus. The official documentation covers the tutorial, language reference, standard library, HOWTOs, FAQs, setup/usage, and more. The Python documentation is licensed under the Python Software Foundation License Version 2; examples, recipes, and other code are additionally licensed under the Zero-Clause BSD License.

Official sources:
- https://docs.python.org/3/
- https://docs.python.org/3/download.html
- https://docs.python.org/3/license.html

The full documentation is substantially larger than the required 20+ pages. This repository does not fabricate a corpus.

## Features

- ~350-word chunks with overlap
- Free local `all-MiniLM-L6-v2` embeddings
- ChromaDB persistent vector store
- Top-5 retrieval
- OpenAI grounded generation
- Page-level source citations
- Out-of-scope refusal behavior
- Streamlit interface

## Setup

```bash
git clone <YOUR_GITHUB_REPO>
cd domain-specific-rag

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Add your PDFs to `data/documents/`.

Set your OpenAI key:

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your_key_here"

# macOS/Linux
export OPENAI_API_KEY="your_key_here"
```

Then build the vector database:

```bash
python -m src.ingest
```

Run:

```bash
streamlit run app.py
```

## Demo Q&A

The examples below are the intended demo questions; run the app after ingestion and record the exact generated answers and source pages.

### 1. In-scope

**Question:** What is a list in Python?

**Expected behavior:** The chatbot retrieves the relevant Python documentation passage and answers with a source citation.

### 2. In-scope

**Question:** How does Python handle exceptions?

**Expected behavior:** The chatbot answers from the relevant Python documentation and cites its source.

### 3. Out-of-scope

**Question:** Who will win the 2027 IPL?

**Expected behavior:**

> I can't answer that from the provided document corpus.

This demonstrates that the chatbot does not blindly use the language model's outside knowledge.

## Evaluation

For the first version, manually test:
- 5 factual questions whose answers are clearly present
- 3 paraphrased versions of those questions
- 3 out-of-scope questions
- 2 questions where the corpus contains related but insufficient information

Record retrieval quality and whether the final answer is grounded.

## Screenshots

Add screenshots here after running the app:

```text
screenshots/chatbot.png
screenshots/out_of_scope.png
```

## Reflection

The main lesson from this project is that RAG quality depends on the full pipeline, not only the language model. Chunk size, overlap, embedding quality, retrieval depth, document quality, and the refusal instruction all affect the final answer.

A limitation of this first version is that it uses simple page-based extraction and fixed-size word chunks. Future versions could add metadata-aware chunking, reranking, retrieval evaluation, conversation memory, and a stronger citation validator.

## Deployment

For Streamlit Community Cloud:

1. Push the repository to GitHub.
2. Create a Streamlit Cloud app pointing to `app.py`.
3. Add `OPENAI_API_KEY` to Streamlit secrets.
4. Ensure the permitted PDF corpus is included in the repository or fetched from an allowed public source.
5. Run the ingestion step before deployment if the vector database is included in the deployment artifact.

Never commit API keys.

## Project checklist

- [ ] 20+ pages of permitted corpus
- [ ] Ingestion completed
- [ ] 5 retrieved chunks verified
- [ ] 3 demo Q&A recorded
- [ ] Out-of-scope refusal tested
- [ ] Screenshot added
- [ ] GitHub repository published
- [ ] Streamlit live URL published
