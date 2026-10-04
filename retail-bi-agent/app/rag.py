"""RAG over the company policy documents.

  docs/*.md  ->  split by heading  ->  embed  ->  in-memory vector store  ->  search tool

Try it from the terminal (no agent involved):
  python -m app.rag "Can I return opened headphones?"
"""
import os
import sys
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter

load_dotenv()

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
HEADERS = [("#", "title"), ("##", "section"), ("###", "subsection")]


# ---------- 1. Chunking ----------
def load_chunks(docs_dir: Path = DOCS_DIR) -> list[Document]:
    """Split every .md file into one chunk per heading section.

    Each chunk starts with 'Document title — section' so it is understandable on its own
    (e.g. a chunk that just says '15% off every product' is useless without knowing it's about Electronics).
    """
    files = sorted(Path(docs_dir).glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No .md files found in {docs_dir}. Did you unzip the docs folder next to app/?")

    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS, strip_headers=True)
    chunks = []
    for path in files:
        for piece in splitter.split_text(path.read_text(encoding="utf-8")):
            body = piece.page_content.strip()
            if not body:
                continue
            title = piece.metadata.get("title", path.stem)
            section = " > ".join(x for x in (piece.metadata.get("section"), piece.metadata.get("subsection")) if x)
            header = f"{title} — {section}" if section else title
            chunks.append(Document(
                page_content=f"{header}\n{body}",
                metadata={"source": path.name, "section": section or "(intro)"},
            ))
    return chunks


# ---------- 2. Embeddings + vector store ----------
class LocalEmbeddings(Embeddings):
    """Free, offline embeddings that run on your own machine (no API key, no cost).

    Uses fastembed + a small model (BAAI/bge-small-en-v1.5, ~70 MB). The model downloads once on first
    use and is cached in .cache/fastembed/ inside the project.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from fastembed import TextEmbedding   # imported here so OpenAI-only users don't need it
        cache_dir = Path(__file__).resolve().parent.parent / ".cache" / "fastembed"
        self._model = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vec.tolist() for vec in self._model.passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(text))).tolist()


def get_embeddings():
    """Embeddings turn text into vectors so 'headphones' can match 'Electronics' by meaning.

    EMBEDDING_PROVIDER=local   (default) free, runs on your laptop
    EMBEDDING_PROVIDER=openai  uses OpenAI's API (needs credits); reads EMBEDDING_API_KEY or OPENAI_API_KEY
    """
    if os.getenv("EMBEDDING_PROVIDER", "local").lower() == "openai":
        return OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            api_key=os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("EMBEDDING_BASE_URL") or None,
            check_embedding_ctx_length=False,   # our chunks are tiny; skips an unnecessary tokenizer download
        )
    return LocalEmbeddings(os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))


def build_vectorstore(embeddings) -> InMemoryVectorStore:
    return InMemoryVectorStore.from_documents(load_chunks(), embeddings)


@lru_cache(maxsize=1)
def get_vectorstore() -> InMemoryVectorStore:
    """Built once per process (about 25 small chunks, so this costs a fraction of a cent)."""
    return build_vectorstore(get_embeddings())


# ---------- 3. Search + the agent tool ----------
def search_policies(query: str, k: int = 3, store=None):
    """Return [(Document, similarity_score), ...] for the k most relevant chunks."""
    store = store or get_vectorstore()
    return store.similarity_search_with_score(query, k=k)


@tool
def search_company_docs(query: str) -> str:
    """Search UrbanCart's internal policy documents.

    Use this for questions about: returns and refunds, discount rules, promotions (and when they
    started or ended), store opening hours, store renovations or closures, and the promotions calendar.
    Also use it to find possible business REASONS behind a change in the numbers.
    Do NOT use it for sales figures, counts or rankings: those come from the database.
    Input: a short natural-language search query.
    """
    results = search_policies(query, k=3)
    if not results:
        return "No relevant policy text found."
    return "\n\n".join(
        f"[source: {doc.metadata['source']} | {doc.metadata['section']}]\n{doc.page_content}"
        for doc, _score in results
    )


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Can I return opened headphones?"
    print(f"Query: {question}\n")
    for doc, score in search_policies(question, k=3):
        print(f"score={score:.3f}  {doc.metadata['source']}  >  {doc.metadata['section']}")
        print(doc.page_content[:300].replace("\n", "\n    "), "\n")
