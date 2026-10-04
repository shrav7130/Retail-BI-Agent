"""RAG tests.

Part 1 (always runs): chunking + search plumbing, using a FAKE word-hashing embedding. Free, offline.
Part 2 (uses the real local model; first run downloads ~70 MB): checks that REAL embeddings
        retrieve the right policy chunk for realistic questions. This is the part that matters.
"""
import math
import os
import re
import zlib

import pytest
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

import app.rag as rag

load_dotenv()


class WordHashEmbeddings(Embeddings):
    """Deterministic bag-of-words vectors. Only for testing plumbing, NOT semantic."""
    DIM = 512

    def _vec(self, text):
        v = [0.0] * self.DIM
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            v[zlib.crc32(w.encode()) % self.DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture(scope="module")
def fake_store():
    return rag.build_vectorstore(WordHashEmbeddings())


# ---------- Part 1: plumbing ----------
def test_chunks_cover_all_documents_and_have_metadata():
    chunks = rag.load_chunks()
    assert len(chunks) >= 20
    assert {c.metadata["source"] for c in chunks} == {
        "discount_policy.md", "return_policy.md", "store_operations.md", "promotions_calendar.md"}
    assert all(c.page_content.strip() and c.metadata["section"] for c in chunks)


def test_promo_chunk_is_self_contained():
    chunk = next(c for c in rag.load_chunks() if "Electronics Clearance" in c.metadata["section"]
                 and c.metadata["source"] == "discount_policy.md")
    assert "31 August 2026" in chunk.page_content and "Discount Policy" in chunk.page_content


def test_search_returns_k_results_with_scores(fake_store):
    results = rag.search_policies("when did the electronics promotion end", k=3, store=fake_store)
    assert len(results) == 3 and all(isinstance(score, float) for _, score in results)


def test_tool_output_includes_sources(fake_store, monkeypatch):
    monkeypatch.setattr(rag, "get_vectorstore", lambda: fake_store)
    out = rag.search_company_docs.invoke({"query": "Pune Camp renovation dates"})
    assert "[source: store_operations.md" in out


def test_tool_has_a_useful_description_for_the_agent():
    assert "returns" in rag.search_company_docs.description.lower()
    assert "database" in rag.search_company_docs.description.lower()


def test_local_embeddings_wrapper_uses_fastembed_correctly(monkeypatch):
    """Checks our wrapper's logic with a stand-in model (no download needed)."""
    import sys
    import types

    import numpy as np

    class FakeTextEmbedding:
        def __init__(self, model_name, cache_dir=None):
            self.model_name = model_name

        def passage_embed(self, texts):
            return (np.array([1.0, 0.0, float(len(t))]) for t in texts)

        def query_embed(self, text):
            return iter([np.array([0.0, 1.0, 9.0])])

    monkeypatch.setitem(sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=FakeTextEmbedding))
    emb = rag.LocalEmbeddings("fake-model")
    docs = emb.embed_documents(["ab", "abcd"])
    assert docs == [[1.0, 0.0, 2.0], [1.0, 0.0, 4.0]] and isinstance(docs[0][0], float)
    assert emb.embed_query("hello") == [0.0, 1.0, 9.0]


def test_default_provider_is_local(monkeypatch):
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    import sys
    import types

    class Fake:
        def __init__(self, model_name, cache_dir=None): pass

    monkeypatch.setitem(sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=Fake))
    assert isinstance(rag.get_embeddings(), rag.LocalEmbeddings)


# ---------- Part 2: real semantic retrieval (downloads the ~70 MB model on first run) ----------
needs_key = pytest.mark.skipif(
    os.getenv("EMBEDDING_PROVIDER", "local").lower() == "openai"
    and not (os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")),
    reason="EMBEDDING_PROVIDER=openai but no API key set",
)


def top_hits(query, k=3):
    return [(d.metadata["source"], d.metadata["section"]) for d, _ in rag.search_policies(query, k=k)]


@needs_key
@pytest.mark.parametrize("query, source, section_part", [
    ("Can I return opened headphones?", "return_policy.md", "Electronics"),
    ("When did the electronics discount stop and why?", "discount_policy.md", "Electronics Clearance"),
    ("Why was the Pune store so quiet in September?", "store_operations.md", "Pune Camp renovation"),
    ("How long do refunds take?", "return_policy.md", "Refunds"),
    ("Was there any promotion running in September?", "promotions_calendar.md", "September 2026"),
])
def test_real_retrieval_finds_the_right_chunk(query, source, section_part):
    hits = top_hits(query, k=3)
    assert any(s == source and section_part in sec for s, sec in hits), f"{query!r} -> {hits}"
