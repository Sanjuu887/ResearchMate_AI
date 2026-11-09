"""MCP-backed research workflow and report serialization."""

import asyncio
import json
import re
from collections.abc import Callable

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

from config import AppConfig, use_config
from researchmate import build_evidence_sources, deduplicate_results, link_citations

DEPTH_SOURCE_LIMITS = {"Quick Answer": 3, "Standard Research": 5, "Deep Research": 8}
DEPTH_VALUES = {
    "Quick Answer": "quick",
    "Standard Research": "standard",
    "Deep Research": "deep",
}


async def call_tool_async(tool_name: str, params: dict, url: str) -> dict:
    transport = StreamableHttpTransport(url)
    async with Client(transport) as client:
        result = await client.call_tool(tool_name, params)
    if getattr(result, "is_error", False):
        raise RuntimeError(f"MCP tool {tool_name} reported an error.")
    content = getattr(result, "content", None) or []
    for item in content:
        text = getattr(item, "text", None)
        if not text:
            continue
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"MCP tool {tool_name} returned an invalid response."
            ) from exc
        if isinstance(data, dict):
            return data
    raise RuntimeError(f"MCP tool {tool_name} returned no usable result.")


def call_tool(tool_name: str, params: dict, url: str) -> dict:
    try:
        return asyncio.run(call_tool_async(tool_name, params, url))
    except (RuntimeError, OSError, TimeoutError) as exc:
        return {"error": f"{tool_name} failed: {exc}"}
    except Exception as exc:
        return {"error": f"{tool_name} failed: {exc}"}


async def call_local_tool_async(
    tool_name: str, params: dict, config: AppConfig
) -> dict:
    import server

    tools = {
        "plan_research": server.plan_research,
        "generate_answer": server.generate_answer,
        "search_web": server.search_web,
        "fetch_and_chunk": server.fetch_and_chunk,
        "cluster_findings": server.cluster_findings,
    }
    tool = tools.get(tool_name)
    if tool is None:
        raise ValueError(f"Unknown local research tool: {tool_name}")
    with use_config(config):
        result = await tool(**params)
    if not isinstance(result, dict):
        raise RuntimeError(f"Local tool {tool_name} returned an invalid response.")
    return result


def call_local_tool(tool_name: str, params: dict, config: AppConfig) -> dict:
    try:
        return asyncio.run(call_local_tool_async(tool_name, params, config))
    except Exception as exc:
        return {"error": f"{tool_name} failed: {exc}"}


def _check_result(result: dict, stage: str) -> dict:
    if result.get("error"):
        raise RuntimeError(f"{stage}: {result['error']}")
    return result


def _cited_sources(answer: str, sources: list[dict]) -> list[dict]:
    cited_ids = {f"S{number}" for number in re.findall(r"\[S(\d+)\]", answer)}
    return [source for source in sources if source.get("id") in cited_ids]


def citation_number(source: dict, fallback: int) -> int:
    try:
        return int(str(source.get("id", "")).removeprefix("S"))
    except ValueError:
        return fallback


def run_research(
    question: str,
    depth: str,
    server_url: str,
    status,
    conversation_history: list[dict] | None = None,
) -> dict:
    return _run_research(
        question,
        depth,
        status,
        lambda name, params: call_tool(name, params, server_url),
        conversation_history,
    )


def run_research_direct(
    question: str,
    depth: str,
    status,
    config: AppConfig,
    conversation_history: list[dict] | None = None,
) -> dict:
    """Run the same research tools in-process, without an MCP server endpoint."""
    return _run_research(
        question,
        depth,
        status,
        lambda name, params: call_local_tool(name, params, config),
        conversation_history,
    )


def _run_research(
    question: str,
    depth: str,
    status,
    call_research_tool: Callable[[str, dict], dict],
    conversation_history: list[dict] | None = None,
) -> dict:
    plan = _check_result(
        call_research_tool(
            "plan_research",
            {
                "question": question,
                "depth": DEPTH_VALUES[depth],
                "conversation_history": conversation_history or [],
            },
        ),
        "Research planning failed",
    )
    if not plan.get("needs_research"):
        answer = plan.get("direct_answer")
        if not isinstance(answer, str) or not answer.strip():
            status.update(label="Generating a direct answer", state="running")
            answer_result = _check_result(
                call_research_tool(
                    "generate_answer",
                    {
                        "question": question,
                        "sources": [],
                        "depth": DEPTH_VALUES[depth],
                        "needs_research": False,
                        "conversation_history": conversation_history or [],
                    },
                ),
                "Answer generation failed",
            )
            answer = answer_result["answer"]
        return {
            "question": question,
            "answer": answer,
            "sources": [],
            "retrieved_count": 0,
            "clusters": [],
            "depth": depth,
            "searched": False,
        }

    queries = plan.get("queries", [])
    if not queries:
        raise RuntimeError("The research planner did not provide any search queries.")

    status.update(label="Searching the web", state="running")
    search_results = []
    for query in queries:
        result = _check_result(
            call_research_tool(
                "search_web",
                {"query": query, "max_results": 6},
            ),
            "Web search failed",
        )
        search_results.extend(result.get("results", []))

    status.update(
        label=f"Search returned {len(search_results)} result(s); filtering relevance",
        state="running",
    )
    relevant_results = deduplicate_results(search_results, question, queries)
    selected_results = relevant_results[: DEPTH_SOURCE_LIMITS[depth]]
    status.update(
        label=f"Fetching content from {len(selected_results)} selected source(s)",
        state="running",
    )
    chunks = []
    fetch_note = None
    if selected_results:
        fetch_result = call_research_tool(
            "fetch_and_chunk",
            {
                "urls": [result["url"] for result in selected_results],
                "chunk_size": 350,
                "chunk_overlap": 40,
                "max_chunks_per_url": 4,
            },
        )
        if fetch_result.get("error"):
            fetch_note = f"Page extraction was unavailable: {fetch_result['error']}"
        else:
            chunks = fetch_result.get("chunks", [])
            failed_pages = fetch_result.get("failed_pages", [])
            if failed_pages:
                fetch_note = (
                    f"{len(failed_pages)} page(s) could not be fetched. "
                    "Their search-result snippets, if available, are labeled below."
                )

    sources = build_evidence_sources(
        selected_results, chunks, DEPTH_SOURCE_LIMITS[depth]
    )
    clusters = []
    status.update(
        label=f"Analyzing evidence from {len(sources)} source(s)",
        state="running",
    )
    if depth == "Deep Research" and len(chunks) >= 2:
        status.update(label="Analyzing themes with lexical clustering", state="running")
        cluster_result = call_research_tool(
            "cluster_findings",
            {"chunks": chunks, "n_clusters": min(4, len(chunks))},
        )
        if cluster_result.get("error"):
            fetch_note = (
                f"{fetch_note} " if fetch_note else ""
            ) + f"Optional TF-IDF analysis was unavailable: {cluster_result['error']}"
        else:
            clusters = cluster_result.get("clusters", [])

    status.update(label="Generating an evidence-based answer", state="running")
    answer_result = _check_result(
        call_research_tool(
            "generate_answer",
            {
                "question": question,
                "sources": sources,
                "depth": DEPTH_VALUES[depth],
                "needs_research": True,
                "conversation_history": conversation_history or [],
            },
        ),
        "Answer generation failed",
    )
    answer = answer_result["answer"]
    used_sources = _cited_sources(answer, sources)
    note = fetch_note
    if sources and not used_sources:
        note = (
            f"{note} " if note else ""
        ) + (
            "Sources were retrieved, but the answer did not cite a supporting "
            "source; they are not presented as verified references."
        )
    return {
        "question": question,
        "answer": answer,
        "sources": used_sources,
        "retrieved_count": len(sources),
        "clusters": clusters,
        "depth": depth,
        "searched": True,
        "note": note,
    }


def make_markdown(question: str, result: dict) -> str:
    sources = result["sources"]
    answer = link_citations(result["answer"], sources)
    lines = [
        "# ResearchMate AI",
        "",
        f"**Question:** {question}",
        f"**Research depth:** {result['depth']}",
        "",
        answer,
    ]
    if result.get("note"):
        lines.extend(["", f"> Note: {result['note']}"])
    if sources:
        lines.extend(["", "## Sources"])
        for index, source in enumerate(sources, 1):
            title = " ".join(source["title"].split())
            title = title.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")
            date = f" — {source['published_date']}" if source.get("published_date") else ""
            provenance = (
                "page content" if source["retrieved_as"] == "page" else "search snippet only"
            )
            lines.append(
                f"{citation_number(source, index)}. [{title}](<{source['url']}>) — "
                f"{provenance}{date}"
            )
    if result.get("clusters"):
        lines.extend(
            [
                "",
                "## Optional lexical grouping",
                "",
                "TF-IDF/K-Means groups shared word usage; these are not semantic findings.",
            ]
        )
        for cluster in result["clusters"]:
            keywords = ", ".join(cluster.get("keywords", []))
            lines.append(f"- **{cluster.get('label', 'Topic')}:** {keywords}")
    return "\n".join(lines).strip()


def make_json(question: str, result: dict, markdown: str) -> str:
    return json.dumps(
        {
            "project": "ResearchMate AI",
            "question": question,
            "depth": result["depth"],
            "searched": result["searched"],
            "answer_markdown": markdown,
            "sources": result["sources"],
            "retrieved_source_count": result.get("retrieved_count", len(result["sources"])),
            "lexical_clusters": result.get("clusters", []),
            "notes": [result["note"]] if result.get("note") else [],
        },
        indent=2,
        ensure_ascii=False,
    )
