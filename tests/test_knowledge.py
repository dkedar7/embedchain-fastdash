"""Knowledge layer: ingestion, retrieval, per-session isolation, loader dispatch."""

from langchain_core.documents import Document

from rag import knowledge


def test_ingest_and_retrieve():
    knowledge.reset("kt1")
    n, label = knowledge.ingest(
        "kt1", "Photosynthesis converts sunlight into chemical energy in plants."
    )
    assert n >= 1
    docs = knowledge.retrieve("kt1", "how do plants make energy?")
    assert docs and "Photosynthesis" in docs[0].page_content


def test_thread_isolation():
    knowledge.reset("kt2")
    knowledge.reset("kt3")
    knowledge.ingest("kt2", "A closely guarded secret.")
    assert knowledge.retrieve("kt3", "secret") == []       # other session sees nothing


def test_reset_clears_store():
    knowledge.ingest("kt5", "something")
    knowledge.reset("kt5")
    assert knowledge.retrieve("kt5", "something") == []


def test_format_passages_numbers_and_tags_sources():
    docs = [Document(page_content="body", metadata={"source": "http://x"})]
    out = knowledge.format_passages(docs)
    assert "[1]" in out and "http://x" in out
    assert "empty" in knowledge.format_passages([]).lower()


def test_url_dispatches_to_web_loader(monkeypatch):
    import langchain_community.document_loaders as loaders

    class FakeWeb:
        def __init__(self, url):
            self.url = url

        def load(self):
            return [Document(page_content="web page body", metadata={"source": self.url})]

    monkeypatch.setattr(loaders, "WebBaseLoader", FakeWeb)
    knowledge.reset("u1")
    n, _ = knowledge.ingest("u1", "https://example.com/article")
    assert n >= 1
    assert "web page body" in knowledge.retrieve("u1", "body")[0].page_content


def test_bad_source_raises_are_left_to_caller(monkeypatch):
    import langchain_community.document_loaders as loaders

    class Boom:
        def __init__(self, url):
            pass

        def load(self):
            raise RuntimeError("network down")

    monkeypatch.setattr(loaders, "WebBaseLoader", Boom)
    import pytest

    with pytest.raises(RuntimeError):
        knowledge.ingest("u2", "https://example.com")   # add_source tool wraps this
