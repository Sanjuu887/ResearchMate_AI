import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx

from config import AppConfig
import researchmate
import server


def _mock_gemini_client(text=None):
    client = Mock()
    async_client = Mock()
    async_client.__aenter__ = AsyncMock(return_value=async_client)
    async_client.__aexit__ = AsyncMock(return_value=None)
    async_client.models = Mock()
    async_client.models.generate_content = AsyncMock(return_value=Mock(text=text))
    client.aio = async_client
    return client, async_client


class GeminiAPIError(Exception):
    def __init__(self, code, status="", message=""):
        super().__init__(message)
        self.code = code
        self.status = status
        self.message = message


def _mock_gemini_sdk(text=None):
    client, async_client = _mock_gemini_client(text)
    genai = Mock(Client=Mock(return_value=client))
    types = Mock(
        GenerateContentConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        HttpOptions=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    return (genai, types, GeminiAPIError), genai, async_client


class ResearchPlanningTests(unittest.IsolatedAsyncioTestCase):
    def test_simple_greetings_and_thanks_get_an_instant_local_response(self):
        for question in ("hello", "Hi!", "hey.", "good morning", "thanks!"):
            with self.subTest(question=question):
                self.assertIsNotNone(researchmate.quick_greeting_response(question))
        self.assertIsNone(
            researchmate.quick_greeting_response("What is photosynthesis?")
        )

    async def test_conversational_greeting_does_not_search(self):
        with patch.object(
            researchmate,
            "_generate_with_gemini",
            new=AsyncMock(
                return_value=(
                    '{"needs_research": false, "queries": [], '
                    '"direct_answer": "Hello! What would you like to explore?"}'
                )
            ),
        ):
            result = await researchmate.plan_question("Hi there!", "quick")
        self.assertEqual(
            result,
            {
                "needs_research": False,
                "queries": [],
                "direct_answer": "Hello! What would you like to explore?",
            },
        )

    async def test_factual_current_comparison_and_deep_questions_can_be_planned(self):
        cases = [
            ("What is photosynthesis?", "quick", ["photosynthesis basics"]),
            ("What is the current inflation rate in Canada?", "standard",
             ["Canada latest inflation rate"]),
            ("Compare PostgreSQL and MongoDB", "standard",
             ["PostgreSQL vs MongoDB use cases"]),
            ("Review the long-term evidence for heat pumps", "deep",
             ["heat pump long term field study", "heat pump operating cost evidence",
              "heat pump limitations research", "extra query"]),
        ]
        for question, depth, queries in cases:
            payload = json.dumps({"needs_research": True, "queries": queries})
            with self.subTest(question=question), patch.object(
                researchmate, "_generate_with_gemini", new=AsyncMock(return_value=payload)
            ):
                plan = await researchmate.plan_question(question, depth)
            self.assertTrue(plan["needs_research"])
            self.assertLessEqual(len(plan["queries"]), researchmate.DEPTH_QUERY_LIMITS[depth])

    async def test_planner_requests_compact_json_with_bounded_direct_answer(self):
        completion = AsyncMock(
            return_value=(
                '{"needs_research": false, "queries": [], '
                '"direct_answer": "A concise answer."}'
            )
        )
        with patch.object(researchmate, "_generate_with_gemini", new=completion):
            await researchmate.plan_question("Explain a stable concept.", "quick")

        kwargs = completion.await_args.kwargs
        self.assertEqual(kwargs["max_tokens"], 1000)
        self.assertTrue(kwargs["retry_on_max_tokens"])
        instructions = completion.await_args.args[0][0]["content"]
        self.assertIn("Return no reasoning or extra fields", instructions)
        self.assertIn("at most 40 words", instructions)

    async def test_planner_rejects_invalid_model_response_and_questions(self):
        with patch.object(
            researchmate, "_generate_with_gemini", new=AsyncMock(return_value="not json")
        ):
            with self.assertRaises(researchmate.ResearchError):
                await researchmate.plan_question("A real question")
        for question in (" ", "x" * (researchmate.MAX_QUESTION_LENGTH + 1)):
            with self.assertRaises(ValueError):
                researchmate.validate_question(question)

    async def test_planner_accepts_fenced_or_prefaced_json(self):
        plan_texts = [
            '```json\n{"needs_research": true, "queries": ["machine learning history"]}\n```',
            'Plan: {"needs_research": true, "queries": ["machine learning history"]}',
        ]
        for raw in plan_texts:
            with self.subTest(raw=raw), patch.object(
                researchmate,
                "_generate_with_gemini",
                new=AsyncMock(return_value=raw),
            ):
                plan = await researchmate.plan_question("Evolution of machine learning")
            self.assertEqual(
                plan,
                {
                    "needs_research": True,
                    "queries": ["machine learning history"],
                },
            )

    async def test_missing_llm_credentials_is_explicit(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            with self.assertRaisesRegex(researchmate.ResearchError, "GEMINI_API_KEY"):
                await researchmate.plan_question("A factual question")

    async def test_gemini_generate_content_uses_configured_model_and_json_mode(self):
        sdk, genai, async_client = _mock_gemini_sdk(
            text='{"needs_research": false}'
        )
        with patch.dict(
            os.environ,
            {"GEMINI_API_KEY": "test-only-key", "GEMINI_MODEL": "gemini-test-model"},
        ), patch.object(
            researchmate, "_load_gemini_sdk", return_value=sdk
        ):
            text = await researchmate._generate_with_gemini(
                [
                    {"role": "system", "content": "Return valid JSON."},
                    {"role": "user", "content": "Plan this question."},
                ],
                json_mode=True,
                max_tokens=123,
            )
        self.assertEqual(text, '{"needs_research": false}')
        genai.Client.assert_called_once()
        self.assertEqual(genai.Client.call_args.kwargs["api_key"], "test-only-key")
        call = async_client.models.generate_content.await_args.kwargs
        self.assertEqual(call["model"], "gemini-test-model")
        self.assertEqual(call["contents"], "Plan this question.")
        self.assertEqual(call["config"].system_instruction, "Return valid JSON.")
        self.assertEqual(call["config"].response_mime_type, "application/json")
        self.assertEqual(call["config"].max_output_tokens, 123)
        self.assertEqual(
            genai.Client.call_args.kwargs["http_options"].timeout, 60_000
        )

    async def test_gemini_quota_error_is_actionable_and_does_not_expose_api_body(self):
        sdk, _, async_client = _mock_gemini_sdk()
        async_client.models.generate_content.side_effect = GeminiAPIError(
            429, "RESOURCE_EXHAUSTED", "private quota detail"
        )
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-only-key"}), patch.object(
            researchmate, "_load_gemini_sdk", return_value=sdk
        ):
            with self.assertRaisesRegex(
                researchmate.ResearchError, "quota or rate limit"
            ) as error:
                await researchmate._generate_with_gemini(
                    [{"role": "system", "content": "System."},
                     {"role": "user", "content": "Question."}]
                )
        self.assertNotIn("private quota detail", str(error.exception))

    async def test_gemini_model_access_error_is_actionable(self):
        sdk, _, async_client = _mock_gemini_sdk()
        async_client.models.generate_content.side_effect = GeminiAPIError(
            404,
            "NOT_FOUND",
            "model not found for key test-secret-key; verify API access",
        )
        with patch.dict(
            os.environ,
            {"GEMINI_API_KEY": "test-secret-key", "GEMINI_MODEL": "unknown-model"},
        ), patch.object(researchmate, "_load_gemini_sdk", return_value=sdk):
            with self.assertRaisesRegex(
                researchmate.ResearchError,
                r"unknown-model.*available to your API key.*verify API access",
            ) as error:
                await researchmate._generate_with_gemini(
                    [{"role": "system", "content": "System."},
                     {"role": "user", "content": "Question."}]
                )
        self.assertNotIn("test-secret-key", str(error.exception))
        self.assertIn("[redacted]", str(error.exception))

    async def test_gemini_bad_request_includes_detail_without_exposing_key(self):
        sdk, _, async_client = _mock_gemini_sdk()
        async_client.models.generate_content.side_effect = GeminiAPIError(
            400,
            "INVALID_ARGUMENT",
            "Invalid request from test-secret-key: unsupported setting",
        )
        with patch.dict(
            os.environ, {"GEMINI_API_KEY": "test-secret-key"}
        ), patch.object(researchmate, "_load_gemini_sdk", return_value=sdk):
            with self.assertRaisesRegex(
                researchmate.ResearchError,
                r"HTTP 400\): Invalid request from \[redacted\]: unsupported setting",
            ) as error:
                await researchmate._generate_with_gemini(
                    [{"role": "system", "content": "System."},
                     {"role": "user", "content": "Question."}]
                )
        self.assertNotIn("test-secret-key", str(error.exception))

    async def test_gemini_default_model_is_catalog_listed_flash_model(self):
        sdk, _, async_client = _mock_gemini_sdk(text="Answer.")
        with patch.object(
            researchmate,
            "get_config",
            return_value=AppConfig(gemini_api_key="test-only-key"),
        ), patch.object(researchmate, "_load_gemini_sdk", return_value=sdk):
            await researchmate._generate_with_gemini(
                [{"role": "system", "content": "System."},
                 {"role": "user", "content": "Question."}]
            )
        self.assertEqual(
            async_client.models.generate_content.await_args.kwargs["model"],
            "gemini-3.8-flash",
        )

    async def test_planner_retries_output_limit_with_larger_budget(self):
        sdk, _, async_client = _mock_gemini_sdk()
        truncated = SimpleNamespace(
            text='{"needs_research": false, "direct_answer":',
            candidates=[SimpleNamespace(finish_reason="MAX_TOKENS")],
        )
        complete = Mock(
            text='{"needs_research": true, "queries": ["AI risk evidence"]}'
        )
        async_client.models.generate_content.side_effect = [truncated, complete]
        with patch.dict(
            os.environ,
            {"GEMINI_API_KEY": "test-only-key", "GEMINI_MODEL": "test-model"},
        ), patch.object(researchmate, "_load_gemini_sdk", return_value=sdk):
            result = await researchmate.plan_question(
                "I said AI is not dangerous, prove me wrong...!!!", "standard"
            )

        self.assertTrue(result["needs_research"])
        self.assertEqual(result["queries"], ["AI risk evidence"])
        calls = async_client.models.generate_content.await_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(
            [call.kwargs["config"].max_output_tokens for call in calls],
            [1000, 2000],
        )

    async def test_gemini_model_setting_accepts_copied_env_assignment_and_list_name(self):
        for configured_model in (
            "GEMINI_MODEL=gemini-3.8-flash",
            "models/gemini-3.8-flash",
        ):
            sdk, _, async_client = _mock_gemini_sdk(text="Answer.")
            with self.subTest(configured_model=configured_model), patch.dict(
                os.environ,
                {
                    "GEMINI_API_KEY": "test-only-key",
                    "GEMINI_MODEL": configured_model,
                },
            ), patch.object(researchmate, "_load_gemini_sdk", return_value=sdk):
                await researchmate._generate_with_gemini(
                    [{"role": "system", "content": "System."},
                     {"role": "user", "content": "Question."}]
                )
            self.assertEqual(
                async_client.models.generate_content.await_args.kwargs["model"],
                "gemini-3.8-flash",
            )

    async def test_gemini_retries_transient_server_error_once(self):
        sdk, _, async_client = _mock_gemini_sdk(text="Answer.")
        async_client.models.generate_content.side_effect = [
            GeminiAPIError(503, "UNAVAILABLE", "temporary overload"),
            Mock(text="Answer."),
        ]
        with patch.dict(
            os.environ, {"GEMINI_API_KEY": "test-only-key"}
        ), patch.object(
            researchmate, "_load_gemini_sdk", return_value=sdk
        ), patch.object(researchmate.asyncio, "sleep", new=AsyncMock()):
            answer = await researchmate._generate_with_gemini(
                [{"role": "system", "content": "System."},
                 {"role": "user", "content": "Question."}]
            )
        self.assertEqual(answer, "Answer.")
        self.assertEqual(async_client.models.generate_content.await_count, 2)

    async def test_gemini_persistent_server_error_reports_status_and_detail(self):
        sdk, _, async_client = _mock_gemini_sdk()
        async_client.models.generate_content.side_effect = GeminiAPIError(
            503, "UNAVAILABLE", "temporary overload"
        )
        with patch.dict(
            os.environ, {"GEMINI_API_KEY": "test-only-key"}
        ), patch.object(
            researchmate, "_load_gemini_sdk", return_value=sdk
        ), patch.object(researchmate.asyncio, "sleep", new=AsyncMock()):
            with self.assertRaisesRegex(
                researchmate.ResearchError,
                r"HTTP 503.*UNAVAILABLE: temporary overload",
            ):
                await researchmate._generate_with_gemini(
                    [{"role": "system", "content": "System."},
                     {"role": "user", "content": "Question."}]
                )
        self.assertEqual(async_client.models.generate_content.await_count, 2)

    async def test_gemini_gateway_timeout_is_not_retried(self):
        sdk, _, async_client = _mock_gemini_sdk()
        async_client.models.generate_content.side_effect = GeminiAPIError(
            504, "DEADLINE_EXCEEDED", "deadline expired"
        )
        with patch.dict(
            os.environ, {"GEMINI_API_KEY": "test-only-key"}
        ), patch.object(researchmate, "_load_gemini_sdk", return_value=sdk):
            with self.assertRaisesRegex(
                researchmate.ResearchError, "timed out.*not retried"
            ):
                await researchmate._generate_with_gemini(
                    [{"role": "system", "content": "System."},
                     {"role": "user", "content": "Question."}]
                )
        self.assertEqual(async_client.models.generate_content.await_count, 1)

    async def test_gemini_network_failure_and_empty_response_are_not_answers(self):
        sdk, _, async_client = _mock_gemini_sdk()
        async_client.models.generate_content.side_effect = httpx.ConnectError("offline")
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-only-key"}), patch.object(
            researchmate, "_load_gemini_sdk", return_value=sdk
        ):
            with self.assertRaisesRegex(researchmate.ResearchError, "Could not reach"):
                await researchmate._generate_with_gemini(
                    [{"role": "system", "content": "System."},
                     {"role": "user", "content": "Question."}]
                )

        sdk, _, _ = _mock_gemini_sdk(text=None)
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-only-key"}), patch.object(
            researchmate, "_load_gemini_sdk", return_value=sdk
        ):
            with self.assertRaisesRegex(researchmate.ResearchError, "returned no text"):
                await researchmate._generate_with_gemini(
                    [{"role": "system", "content": "System."},
                     {"role": "user", "content": "Question."}]
                )

    async def test_gemini_timeout_is_reported_without_fallback_answer(self):
        sdk, _, async_client = _mock_gemini_sdk()
        async_client.models.generate_content.side_effect = httpx.ReadTimeout("slow")
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-only-key"}), patch.object(
            researchmate, "_load_gemini_sdk", return_value=sdk
        ):
            with self.assertRaisesRegex(researchmate.ResearchError, "timed out"):
                await researchmate._generate_with_gemini(
                    [{"role": "system", "content": "System."},
                     {"role": "user", "content": "Question."}]
                )


class EvidenceHandlingTests(unittest.TestCase):
    def test_search_results_are_deduplicated_and_unrelated_results_filtered(self):
        results = [
            {
                "title": "PostgreSQL overview",
                "url": "https://example.com/postgres/?utm_source=search",
                "snippet": "PostgreSQL relational database features",
                "score": 0.8,
            },
            {
                "title": "Postgres duplicate",
                "url": "https://EXAMPLE.com/postgres/#section",
                "snippet": "PostgreSQL database system",
                "score": 0.7,
            },
            {
                "title": "Gardening with tomatoes",
                "url": "https://garden.example/articles",
                "snippet": "A guide to growing tomatoes in a small garden.",
                "score": 0.1,
            },
            {
                "title": "Not a web source",
                "url": "file:///private/data",
                "snippet": "PostgreSQL",
                "score": 1,
            },
        ]
        filtered = researchmate.deduplicate_results(
            results,
            "Compare PostgreSQL and MongoDB",
            ["PostgreSQL vs MongoDB"],
        )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["score"], 0.8)

    def test_fetched_content_and_snippet_provenance_are_kept_distinct(self):
        sources = researchmate.build_evidence_sources(
            [
                {
                    "title": "Fetched",
                    "url": "https://example.com/article",
                    "snippet": "Search result excerpt",
                    "published_date": "2025-03-01",
                },
                {
                    "title": "Not fetched",
                    "url": "https://example.org/other",
                    "snippet": "Only the result snippet is available.",
                },
            ],
            [
                {
                    "url": "https://example.com/article#part",
                    "text": "Fetched page text",
                }
            ],
            limit=5,
        )
        self.assertEqual(sources[0]["retrieved_as"], "page")
        self.assertEqual(sources[0]["content"], "Fetched page text")
        self.assertEqual(sources[0]["published_date"], "2025-03-01")
        self.assertEqual(sources[1]["retrieved_as"], "search snippet")
        self.assertEqual(sources[1]["content"], "Only the result snippet is available.")

    def test_only_real_source_citations_are_linked(self):
        sources = [
            {"id": "S1", "url": "https://example.com/source"},
            {"id": "S2", "url": "javascript:alert(1)"},
        ]
        result = researchmate.link_citations(
            "Supported [S1]; unsupported [S9]. "
            "[fake](https://fake.example) [S1](https://wrong.example) "
            "also https://invented.example.",
            sources,
        )
        self.assertIn("[1](<https://example.com/source>)", result)
        self.assertNotIn("[S9]", result)
        self.assertNotIn("https://fake.example", result)
        self.assertNotIn("javascript:", result)


class SynthesisTests(unittest.IsolatedAsyncioTestCase):
    async def test_answer_output_limits_scale_with_research_depth(self):
        cases = [("quick", 2500), ("standard", 4000), ("deep", 10_000)]
        for depth, expected_limit in cases:
            completion = AsyncMock(return_value="A source-supported answer [S1].")
            with self.subTest(depth=depth), patch.object(
                researchmate, "_generate_with_gemini", new=completion
            ):
                await researchmate.synthesize_answer(
                    "What are recent developments?",
                    [{"id": "S1", "content": "Current evidence", "url": "https://example.com"}],
                    depth,
                )
            self.assertEqual(
                completion.await_args.kwargs["max_tokens"], expected_limit
            )

    async def test_standard_research_keeps_requested_lists_concise_and_evidence_bound(self):
        completion = AsyncMock(return_value="A concise, source-supported list [S1].")
        with patch.object(researchmate, "_generate_with_gemini", new=completion):
            await researchmate.synthesize_answer(
                "List 10 current developments",
                [{"id": "S1", "content": "Current evidence", "url": "https://example.com"}],
                "standard",
            )
        instructions = completion.await_args.args[0][0]["content"]
        self.assertIn("Honor requested list counts", instructions)
        self.assertIn("keep each item concise", instructions)
        self.assertIn("do not pad with unsupported claims", instructions)

    async def test_non_research_answer_can_be_conversational(self):
        completion = AsyncMock(return_value="Hello! I can help with that.")
        with patch.object(researchmate, "_generate_with_gemini", new=completion):
            answer = await researchmate.synthesize_answer(
                "Hello!", [], "quick", needs_research=False
            )
        self.assertIn("Hello", answer)
        instructions = completion.await_args.args[0][0]["content"]
        self.assertIn("No web research was requested", instructions)
        self.assertNotIn("using only the supplied evidence", instructions)

    async def test_empty_evidence_is_explicitly_passed_for_insufficient_answer(self):
        completion = AsyncMock(return_value="The available evidence is insufficient.")
        with patch.object(researchmate, "_generate_with_gemini", new=completion):
            answer = await researchmate.synthesize_answer(
                "What caused the event?", [], "standard"
            )
        self.assertIn("insufficient", answer)
        messages = completion.await_args.args[0]
        self.assertIn("missing", messages[0]["content"])
        self.assertEqual(json.loads(messages[1]["content"])["sources"], [])

    async def test_conflicting_sources_are_supplied_for_comparison(self):
        sources = [
            {
                "id": "S1",
                "url": "https://one.example/report",
                "content": "The study estimates savings of 10%.",
                "retrieved_as": "page",
            },
            {
                "id": "S2",
                "url": "https://two.example/report",
                "content": "The study estimates savings of 4%.",
                "retrieved_as": "search snippet",
            },
        ]
        completion = AsyncMock(
            return_value="Estimates differ: 10% [S1] and 4% [S2]."
        )
        with patch.object(researchmate, "_generate_with_gemini", new=completion):
            answer = await researchmate.synthesize_answer(
                "Compare the reported estimates", sources, "deep"
            )
        self.assertIn("conflicting", completion.await_args.args[0][0]["content"])
        self.assertIn("10%", completion.await_args.args[0][1]["content"])
        self.assertIn("4%", completion.await_args.args[0][1]["content"])
        self.assertEqual(answer.count("[S"), 2)


class PageExtractionTests(unittest.IsolatedAsyncioTestCase):
    async def test_legacy_report_markdown_and_json_formats_remain_available(self):
        clusters = [
            {
                "cluster_id": 0,
                "label": "Cost",
                "keywords": ["cost", "energy"],
                "summary": "The evidence discusses energy costs.",
                "chunk_count": 1,
                "source_urls": ["https://example.com/report"],
                "chunks": [
                    {
                        "url": "https://example.com/report",
                        "title": "Example report",
                        "text": "Evidence about energy costs and system operation.",
                    }
                ],
            }
        ]
        markdown = await server.generate_report(
            "Energy costs", clusters, format="markdown"
        )
        json_report = await server.generate_report(
            "Energy costs", clusters, format="json"
        )
        self.assertIn("# ResearchMate AI", markdown["report"])
        self.assertIn("https://example.com/report", markdown["report"])
        self.assertEqual(json.loads(json_report["report"])["sources"][0]["url"],
                         "https://example.com/report")
        invalid = await server.generate_report(
            "Energy costs", clusters, format="pdf"
        )
        self.assertIn("format must", invalid["error"])

    async def test_html_cleanup_removes_navigation_and_decodes_entities(self):
        paragraph = "Evidence &amp; findings from the research. " * 20
        text = (
            "<html><head><title>Study &amp; Results</title></head>"
            "<body><nav>Home Contact Menu</nav><main><p>"
            + paragraph
            + "</p><script>ignore this content</script></main></body></html>"
        )
        title, content = server._extract_page_text(text, "https://example.com")
        self.assertEqual(title, "Study & Results")
        self.assertNotIn("Home Contact", content)
        self.assertNotIn("ignore this content", content)
        self.assertIn("Evidence & findings", content)

    async def test_inaccessible_page_is_returned_as_a_fetch_failure(self):
        class FailedStream:
            async def __aenter__(self):
                raise httpx.ConnectError("offline")

            async def __aexit__(self, exc_type, exc, traceback):
                return None

        class FailedClient:
            def stream(self, method, url):
                return FailedStream()

        result = await server._fetch_one(FailedClient(), "https://example.com/page")
        self.assertIn("Could not fetch", result["error"])

    async def test_missing_tavily_key_and_empty_results_are_explicit(self):
        with patch.object(server, "get_config", return_value=AppConfig()):
            result = await server.search_web("current climate figures")
        self.assertIn("TAVILY_API_KEY", result["error"])
        self.assertEqual(result["success"], False)

    async def test_tavily_empty_results_are_a_valid_empty_search(self):
        response = Mock()
        response.json.return_value = {"results": [], "answer": ""}
        fake_client = AsyncMock()
        fake_client.__aenter__.return_value = fake_client
        fake_client.__aexit__.return_value = None
        fake_client.post.return_value = response
        with patch.object(
            server,
            "get_config",
            return_value=AppConfig(tavily_api_key="test-only-key"),
        ), patch(
            "server.httpx.AsyncClient", return_value=fake_client
        ):
            result = await server.search_web("current climate figures")
        request = fake_client.post.await_args
        self.assertEqual(
            request.kwargs["headers"]["Authorization"], "Bearer test-only-key"
        )
        self.assertNotIn("api_key", request.kwargs["json"])
        self.assertEqual(request.kwargs["json"]["query"], "current climate figures")
        self.assertEqual(result["total_found"], 0)
        self.assertEqual(result["results"], [])

    async def test_tavily_unauthorized_response_has_actionable_sanitized_error(self):
        response = Mock(status_code=401)
        request = httpx.Request("POST", "https://api.tavily.com/search")
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Unauthorized: test-only-key", request=request, response=response
        )
        fake_client = AsyncMock()
        fake_client.__aenter__.return_value = fake_client
        fake_client.__aexit__.return_value = None
        fake_client.post.return_value = response
        with patch.object(
            server,
            "get_config",
            return_value=AppConfig(tavily_api_key="test-only-key"),
        ), patch("server.httpx.AsyncClient", return_value=fake_client):
            result = await server.search_web("current climate figures")

        self.assertIn("Tavily rejected the configured API key", result["error"])
        self.assertIn(".env", result["error"])
        self.assertNotIn("test-only-key", result["error"])

    async def test_tavily_malformed_response_is_reported(self):
        response = Mock()
        response.json.return_value = []
        fake_client = AsyncMock()
        fake_client.__aenter__.return_value = fake_client
        fake_client.__aexit__.return_value = None
        fake_client.post.return_value = response
        with patch.object(
            server,
            "get_config",
            return_value=AppConfig(tavily_api_key="test-only-key"),
        ), patch(
            "server.httpx.AsyncClient", return_value=fake_client
        ):
            result = await server.search_web("current climate figures")
        self.assertIn("malformed", result["error"])


if __name__ == "__main__":
    unittest.main()
