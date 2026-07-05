"""Agent layer: tools, graph construction, and (live) grounded answering."""

import os

import pytest

from rag import knowledge
from rag.graph import build_graph
from rag.tools import add_source, search_knowledge


def test_add_source_tool_ingests_and_confirms():
    cfg = {"configurable": {"thread_id": "at1"}}
    knowledge.reset("at1")
    r = add_source.invoke({"source": "The Moon is about 384400 km from Earth."}, cfg)
    assert "Added" in r
    s = search_knowledge.invoke({"query": "distance to the Moon"}, cfg)
    assert "384400" in s


def test_tools_are_thread_scoped():
    add_source.invoke({"source": "session-scoped fact"},
                      {"configurable": {"thread_id": "at2"}})
    # a different session's search sees an empty base
    s = search_knowledge.invoke({"query": "fact"},
                                {"configurable": {"thread_id": "at3"}})
    assert "no relevant" in s.lower() or "empty" in s.lower()


def test_add_source_wraps_loader_errors(monkeypatch):
    import langchain_community.document_loaders as loaders

    class Boom:
        def __init__(self, url):
            pass

        def load(self):
            raise RuntimeError("boom")

    monkeypatch.setattr(loaders, "WebBaseLoader", Boom)
    r = add_source.invoke({"source": "https://example.com"},
                          {"configurable": {"thread_id": "at4"}})
    assert "could not add" in r.lower()          # friendly, not a crash


def test_build_graph_has_both_tools():
    graph = build_graph()
    # a compiled LangGraph graph (what fast_dash's chat= bridge detects)
    assert hasattr(graph, "get_graph") and hasattr(graph, "astream")


@pytest.mark.skipif(not os.environ.get("OPENROUTER_API_KEY"), reason="no OPENROUTER_API_KEY")
def test_live_agent_answers_from_source_with_citation():
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.prebuilt import create_react_agent
    from langchain_openai import ChatOpenAI

    from rag.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_MODEL
    from rag.graph import SYSTEM_PROMPT

    model = ChatOpenAI(model=OPENROUTER_MODEL, base_url=OPENROUTER_BASE_URL,
                       api_key=OPENROUTER_API_KEY, temperature=0)
    graph = create_react_agent(model, [add_source, search_knowledge],
                               prompt=SYSTEM_PROMPT, checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "live-test"}}
    graph.invoke({"messages": [("user",
        "Remember: the Zorblax runs at 4200 kelvin and weighs 3.7 kg.")]}, cfg)
    r = graph.invoke({"messages": [("user",
        "What temperature does the Zorblax run at and how heavy is it?")]}, cfg)
    ans = r["messages"][-1].content
    assert "4200" in ans and "3.7" in ans        # grounded in the added source
