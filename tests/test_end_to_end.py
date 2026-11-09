import json
import unittest
from unittest.mock import AsyncMock, patch

from fastmcp import Client

from config import AppConfig, get_config
import research_workflow
import server
from report_exports import make_report_pdf


def feature_case(inputs: dict, expected_output: dict):
    def decorate(test_method):
        test_method.feature_case = {
            "inputs": inputs,
            "expected_output": expected_output,
        }
        return test_method

    return decorate


class ResearchWorkflowEndToEndTests(unittest.TestCase):
    def test_direct_quick_answer_skips_search(self):
        config = AppConfig(gemini_api_key="test-key", tavily_api_key="test-key")
        status_updates = []

        class Status:
            def update(self, **kwargs):
                status_updates.append(kwargs)

        with patch.object(
            server,
            "plan_research",
            new=AsyncMock(
                return_value={
                    "needs_research": False,
                    "direct_answer": "Photosynthesis converts light into stored energy.",
                }
            ),
        ) as plan, patch.object(
            server, "search_web", new=AsyncMock()
        ) as search, patch.object(
            server, "generate_answer", new=AsyncMock()
        ) as answer:
            result = research_workflow.run_research_direct(
                "What is photosynthesis?", "Quick Answer", Status(), config
            )

        self.assertFalse(result["searched"])
        self.assertEqual(result["sources"], [])
        self.assertEqual(plan.await_args.kwargs["depth"], "quick")
        search.assert_not_awaited()
        answer.assert_not_awaited()
        self.assertEqual(status_updates, [])

    def test_direct_workflow_runs_without_an_mcp_endpoint(self):
        question = "What are recent battery storage developments?"
        source = {
            "title": "Battery storage research",
            "url": "https://research.example/battery-storage",
            "snippet": "Recent battery storage developments.",
            "score": 0.9,
        }
        chunk = {
            "id": "battery-0",
            "url": source["url"],
            "title": source["title"],
            "text": "Battery storage capacity increased in recent deployments.",
        }
        settings = AppConfig(
            gemini_api_key="test-gemini-key",
            tavily_api_key="test-tavily-key",
        )
        observed_config = []

        async def plan_research(**_params):
            observed_config.append(get_config())
            return {"needs_research": True, "queries": ["battery storage developments"]}

        class Status:
            def update(self, **_kwargs):
                return None

        with patch.object(server, "plan_research", new=AsyncMock(side_effect=plan_research)), patch.object(
            server,
            "search_web",
            new=AsyncMock(return_value={"results": [source]}),
        ), patch.object(
            server,
            "fetch_and_chunk",
            new=AsyncMock(return_value={"chunks": [chunk], "failed_pages": []}),
        ), patch.object(
            server,
            "generate_answer",
            new=AsyncMock(return_value={"answer": "Storage capacity increased [S1]."}),
        ), patch.object(
            research_workflow,
            "call_tool",
            side_effect=AssertionError("Direct execution must not use an MCP client."),
        ):
            result = research_workflow.run_research_direct(
                question, "Standard Research", Status(), settings
            )

        self.assertTrue(result["searched"])
        self.assertEqual(result["sources"][0]["id"], "S1")
        self.assertIs(observed_config[0], settings)

    def test_direct_deep_research_runs_clustering_and_preserves_citations(self):
        queries = ["solar research", "wind research"]
        results = [
            {
                "title": f"{technology.title()} energy research",
                "url": f"https://research.example/{technology}",
                "snippet": f"Recent {technology} energy research.",
                "score": 0.9,
            }
            for technology in ("solar", "wind")
        ]
        chunks = [
            {
                "id": f"{technology}-0",
                "url": results[index]["url"],
                "title": results[index]["title"],
                "text": f"{technology.title()} energy capacity grew.",
            }
            for index, technology in enumerate(("solar", "wind"))
        ]
        config = AppConfig(gemini_api_key="test-key", tavily_api_key="test-key")

        class Status:
            def update(self, **_kwargs):
                return None

        with patch.object(
            server,
            "plan_research",
            new=AsyncMock(return_value={"needs_research": True, "queries": queries}),
        ) as plan, patch.object(
            server,
            "search_web",
            new=AsyncMock(
                side_effect=[
                    {"results": [results[0]]},
                    {"results": [results[1]]},
                ]
            ),
        ) as search, patch.object(
            server,
            "fetch_and_chunk",
            new=AsyncMock(return_value={"chunks": chunks, "failed_pages": []}),
        ), patch.object(
            server,
            "cluster_findings",
            new=AsyncMock(return_value={"clusters": [{"label": "Energy"}]}),
        ) as cluster, patch.object(
            server,
            "generate_answer",
            new=AsyncMock(return_value={"answer": "Solar [S1] and wind [S2]."}),
        ) as answer:
            result = research_workflow.run_research_direct(
                "Compare solar and wind research",
                "Deep Research",
                Status(),
                config,
            )

        self.assertEqual(plan.await_args.kwargs["depth"], "deep")
        self.assertEqual(search.await_count, 2)
        self.assertEqual(cluster.await_args.kwargs["n_clusters"], 2)
        self.assertEqual(answer.await_args.kwargs["depth"], "deep")
        self.assertEqual([source["id"] for source in result["sources"]], ["S1", "S2"])
        self.assertEqual(result["clusters"], [{"label": "Energy"}])

    @feature_case(
        inputs={
            "question": "Compare solar and wind energy research",
            "mode": "Deep Research",
            "fake_search_queries": 2,
            "fake_sources": 2,
            "providers": "mocked",
        },
        expected_output={
            "called_tools": [
                "plan_research",
                "search_web",
                "search_web",
                "fetch_and_chunk",
                "cluster_findings",
                "generate_answer",
            ],
            "cited_source_ids": ["S1", "S2"],
            "cluster_count": 1,
            "markdown": True,
            "json_source_count": 2,
            "pdf_signature": "%PDF-",
        },
    )
    def test_deep_research_runs_search_fetch_cluster_citations_and_all_exports(self):
        question = "Compare solar and wind energy research"
        queries = ["solar energy evidence", "wind energy evidence"]
        search_results = {
            queries[0]: [
                {
                    "title": "Solar energy study",
                    "url": "https://research.example/solar",
                    "snippet": "Solar energy evidence and capacity.",
                    "score": 0.9,
                }
            ],
            queries[1]: [
                {
                    "title": "Wind energy study",
                    "url": "https://research.example/wind",
                    "snippet": "Wind energy evidence and capacity.",
                    "score": 0.9,
                }
            ],
        }
        chunks = [
            {
                "id": "solar-0",
                "url": "https://research.example/solar",
                "text": "Solar capacity grew in the latest reporting period.",
                "title": "Solar energy study",
            },
            {
                "id": "wind-0",
                "url": "https://research.example/wind",
                "text": "Wind capacity grew in the latest reporting period.",
                "title": "Wind energy study",
            },
        ]
        cluster = {
            "label": "Renewable Energy",
            "keywords": ["capacity", "energy"],
            "chunk_count": 2,
            "chunks": chunks,
            "source_urls": [
                "https://research.example/solar",
                "https://research.example/wind",
            ],
        }
        answer = (
            "# Executive summary\n\n"
            "Solar capacity grew [S1], and wind capacity grew [S2]."
        )
        calls = []

        def fake_call_tool(name, params, url):
            calls.append({"tool": name, "input": params, "server_url": url})
            if name == "plan_research":
                return {"needs_research": True, "queries": queries}
            if name == "search_web":
                return {"results": search_results[params["query"]]}
            if name == "fetch_and_chunk":
                return {"chunks": chunks, "failed_pages": []}
            if name == "cluster_findings":
                return {"clusters": [cluster]}
            if name == "generate_answer":
                return {"answer": answer}
            raise AssertionError(f"Unexpected tool call: {name}")

        class Status:
            def __init__(self):
                self.updates = []

            def update(self, **kwargs):
                self.updates.append(kwargs)

        status = Status()
        with patch.object(research_workflow, "call_tool", side_effect=fake_call_tool):
            result = research_workflow.run_research(
                question,
                "Deep Research",
                "http://mcp.test/mcp",
                status,
                [{"role": "user", "content": "Compare renewable options"}],
            )

        self.assertEqual(
            [call["tool"] for call in calls],
            [
                "plan_research",
                "search_web",
                "search_web",
                "fetch_and_chunk",
                "cluster_findings",
                "generate_answer",
            ],
        )
        self.assertEqual(calls[0]["input"]["depth"], "deep")
        self.assertEqual(calls[0]["input"]["conversation_history"][0]["role"], "user")
        self.assertEqual(calls[-1]["input"]["depth"], "deep")
        self.assertEqual(calls[-1]["input"]["needs_research"], True)
        self.assertEqual(result["question"], question)
        self.assertTrue(result["searched"])
        self.assertEqual(result["retrieved_count"], 2)
        self.assertEqual([source["id"] for source in result["sources"]], ["S1", "S2"])
        self.assertEqual([source["retrieved_as"] for source in result["sources"]], ["page", "page"])
        self.assertEqual(result["clusters"], [cluster])
        self.assertIsNone(result["note"])
        self.assertTrue(status.updates)

        markdown = research_workflow.make_markdown(question, result)
        self.assertIn("[1](<https://research.example/solar>)", markdown)
        self.assertIn("[2](<https://research.example/wind>)", markdown)
        self.assertIn("Optional lexical grouping", markdown)
        exported_json = json.loads(
            research_workflow.make_json(question, result, markdown)
        )
        self.assertEqual(exported_json["question"], question)
        self.assertTrue(exported_json["searched"])
        self.assertEqual(exported_json["retrieved_source_count"], 2)
        self.assertEqual(len(exported_json["lexical_clusters"]), 1)

        pdf = make_report_pdf(question, {**result, "generated_at": "2026-11-09T00:00:00+00:00"})
        self.assertTrue(pdf.startswith(b"%PDF-"))
        self.assertTrue(pdf.rstrip().endswith(b"%%EOF"))
        self.feature_output = {
            "called_tools": [call["tool"] for call in calls],
            "cited_source_ids": [source["id"] for source in result["sources"]],
            "cluster_count": len(result["clusters"]),
            "markdown": "[1](<https://research.example/solar>)" in markdown,
            "json_source_count": exported_json["retrieved_source_count"],
            "pdf_signature": pdf[:5].decode("ascii"),
        }

    @feature_case(
        inputs={
            "question": "What is photosynthesis?",
            "mode": "Quick Answer",
            "conversation_turns": 1,
        },
        expected_output={
            "searched": False,
            "source_count": 0,
            "called_tools": ["plan_research"],
        },
    )
    def test_direct_answer_skips_web_tools_and_preserves_chat_history(self):
        question = "What is photosynthesis?"
        history = [{"role": "user", "content": "Explain it simply"}]
        calls = []

        def fake_call_tool(name, params, _url):
            calls.append((name, params))
            if name == "plan_research":
                return {
                    "needs_research": False,
                    "direct_answer": "Plants use light to make sugars.",
                }
            raise AssertionError(f"Unexpected tool call: {name}")

        with patch.object(research_workflow, "call_tool", side_effect=fake_call_tool):
            result = research_workflow.run_research(
                question, "Quick Answer", "http://mcp.test/mcp", object(), history
            )

        self.assertEqual([name for name, _ in calls], ["plan_research"])
        self.assertEqual(calls[0][1]["conversation_history"], history)
        self.assertEqual(
            result,
            {
                "question": question,
                "answer": "Plants use light to make sugars.",
                "sources": [],
                "retrieved_count": 0,
                "clusters": [],
                "depth": "Quick Answer",
                "searched": False,
            },
        )
        self.feature_output = {
            "answer": result["answer"],
            "searched": result["searched"],
            "source_count": len(result["sources"]),
            "called_tools": [name for name, _ in calls],
        }

    @feature_case(
        inputs={
            "question": "What is recent solar capacity?",
            "mode": "Standard Research",
            "page_fetch": "simulated failure",
            "search_snippet": "available",
        },
        expected_output={
            "source_provenance": "search snippet",
            "has_limitation_note": True,
            "called_tools": [
                "plan_research",
                "search_web",
                "fetch_and_chunk",
                "generate_answer",
            ],
        },
    )
    def test_failed_page_fetch_uses_labeled_snippet_and_reports_limitation(self):
        search_result = {
            "title": "Solar report",
            "url": "https://research.example/solar",
            "snippet": "A report about solar capacity and recent energy.",
            "score": 0.9,
        }
        answer = "The search snippet reports recent solar capacity [S1]."
        calls = []

        class Status:
            def update(self, **_kwargs):
                return None

        def fake_call_tool(name, params, _url):
            calls.append(name)
            if name == "plan_research":
                return {"needs_research": True, "queries": ["recent solar capacity"]}
            if name == "search_web":
                return {"results": [search_result]}
            if name == "fetch_and_chunk":
                return {
                    "chunks": [],
                    "failed_pages": [
                        {"url": search_result["url"], "error": "connection timeout"}
                    ],
                }
            if name == "generate_answer":
                self.assertEqual(params["sources"][0]["retrieved_as"], "search snippet")
                return {"answer": answer}
            raise AssertionError(f"Unexpected tool call: {name}")

        with patch.object(research_workflow, "call_tool", side_effect=fake_call_tool):
            result = research_workflow.run_research(
                "What is recent solar capacity?",
                "Standard Research",
                "http://mcp.test/mcp",
                Status(),
            )

        self.assertEqual(
            calls, ["plan_research", "search_web", "fetch_and_chunk", "generate_answer"]
        )
        self.assertEqual(result["sources"][0]["retrieved_as"], "search snippet")
        self.assertIn("could not be fetched", result["note"])
        self.assertIn("search-result snippets", result["note"])
        self.assertIn(
            "search snippet only",
            research_workflow.make_markdown(result["question"], result),
        )
        self.feature_output = {
            "source_provenance": result["sources"][0]["retrieved_as"],
            "has_limitation_note": bool(result["note"]),
            "called_tools": calls,
        }

    @feature_case(
        inputs={
            "question": "What is current?",
            "mode": "Quick Answer",
            "planning_response": "Gemini model unavailable",
        },
        expected_output={"error_stage": "Research planning failed"},
    )
    def test_research_tool_error_is_reported_with_its_workflow_stage(self):
        with patch.object(
            research_workflow,
            "call_tool",
            return_value={"error": "Gemini model is unavailable"},
        ):
            with self.assertRaisesRegex(
                RuntimeError, "Research planning failed: Gemini model is unavailable"
            ):
                research_workflow.run_research(
                    "What is current?",
                    "Quick Answer",
                    "http://mcp.test/mcp",
                    object(),
                )
        self.feature_output = {"error_stage": "Research planning failed"}


class AdditionalMCPFeatureTests(unittest.IsolatedAsyncioTestCase):
    @feature_case(
        inputs={"mcp_transport": "in-memory", "tool": "cluster_findings", "chunks": 2},
        expected_output={
            "registered_tools": 6,
            "tool_call_error": False,
            "clustered_chunks": 2,
        },
    )
    async def test_mcp_transport_lists_tools_and_dispatches_clustering(self):
        chunks = [
            {
                "id": "solar-0",
                "url": "https://research.example/solar",
                "text": "Solar energy power generation and clean electricity.",
            },
            {
                "id": "wind-0",
                "url": "https://research.example/wind",
                "text": "Solar energy power generation and clean electricity.",
            },
        ]
        async with Client(server.mcp) as client:
            tools = await client.list_tools()
            result = await client.call_tool(
                "cluster_findings",
                {"chunks": chunks, "n_clusters": 1},
            )

        payload = json.loads(result.content[0].text)
        self.assertEqual(
            {tool.name for tool in tools},
            {
                "plan_research",
                "generate_answer",
                "search_web",
                "fetch_and_chunk",
                "cluster_findings",
                "generate_report",
            },
        )
        self.assertFalse(result.is_error)
        self.assertEqual(payload["total_chunks_clustered"], 2)
        self.feature_output = {
            "registered_tools": len(tools),
            "tool_call_error": result.is_error,
            "clustered_chunks": payload["total_chunks_clustered"],
        }

    @feature_case(
        inputs={"chunks": 2, "cluster_count": 1},
        expected_output={"cluster_count": 1, "chunk_count": 2},
    )
    async def test_clustering_preserves_every_chunk_and_returns_schema(self):
        chunks = [
            {
                "id": "solar-0",
                "url": "https://research.example/solar",
                "text": "Solar energy power generation and clean electricity.",
            },
            {
                "id": "wind-0",
                "url": "https://research.example/wind",
                "text": "Wind energy power generation and clean electricity.",
            },
        ]

        result = await server.cluster_findings(chunks, n_clusters=1)

        self.assertNotIn("error", result)
        self.assertEqual(result["total_chunks_clustered"], 2)
        self.assertEqual(result["n_clusters"], 1)
        self.assertEqual(len(result["clusters"]), 1)
        self.assertEqual(
            {chunk["id"] for chunk in result["clusters"][0]["chunks"]},
            {"solar-0", "wind-0"},
        )
        self.assertTrue(result["clusters"][0]["keywords"])
        self.feature_output = {
            "cluster_count": result["n_clusters"],
            "chunk_count": result["total_chunks_clustered"],
            "keywords": result["clusters"][0]["keywords"],
        }

    @feature_case(
        inputs={"url": "https://research.example/solar", "mocked_html": True},
        expected_output={
            "fetched_urls": 1,
            "failed_pages": 0,
            "chunk_count_minimum": 1,
        },
    )
    async def test_fetch_tool_returns_chunks_for_mocked_html_page(self):
        page_text = ("Solar energy research reports show capacity increased. " * 35)

        class Response:
            status_code = 200
            headers = {"content-type": "text/html; charset=utf-8"}
            encoding = "utf-8"

            async def aiter_bytes(self):
                yield (
                    "<html><head><title>Solar study</title></head><body><main><p>"
                    + page_text
                    + "</p></main></body></html>"
                ).encode()

            def raise_for_status(self):
                return None

        class Stream:
            async def __aenter__(self):
                return Response()

            async def __aexit__(self, exc_type, exc, traceback):
                return None

        class Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                return None

            def stream(self, _method, _url):
                return Stream()

        with patch("server.httpx.AsyncClient", return_value=Client()):
            result = await server.fetch_and_chunk(
                ["https://research.example/solar"],
                chunk_size=100,
                chunk_overlap=10,
                max_chunks_per_url=2,
            )

        self.assertEqual(result["fetched_urls"], 1)
        self.assertEqual(result["failed_pages"], [])
        self.assertGreaterEqual(result["total_chunks"], 1)
        self.assertTrue(all(chunk["url"] == "https://research.example/solar" for chunk in result["chunks"]))
        self.assertIn("Solar energy research", result["chunks"][0]["text"])
        self.feature_output = {
            "fetched_urls": result["fetched_urls"],
            "failed_pages": len(result["failed_pages"]),
            "chunk_count": result["total_chunks"],
            "page_title": result["chunks"][0]["title"],
        }


if __name__ == "__main__":
    unittest.main()
