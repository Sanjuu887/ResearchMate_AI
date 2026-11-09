# ResearchMate AI

A Streamlit research assistant that searches the web, summarizes retrieved evidence with Gemini, and links claims to sources. The Streamlit app runs the research workflow directly; a separate FastMCP server remains available for local MCP clients.

## Project overview

ResearchMate AI helps investigate questions that benefit from current web evidence. It plans focused searches, retrieves and cleans accessible pages, labels search snippets when pages cannot be fetched, and creates reports with citations.

Generated answers can be wrong. Check important claims against the linked sources.

## Features

- Gemini-powered question planning and evidence-based answers.
- Tavily web search, page extraction, source filtering, and citations.
- Optional TF-IDF/K-Means grouping in Deep Research.
- Markdown, JSON, and PDF report downloads.
- Local FastMCP tools for MCP-compatible clients.

## Research modes

- **Quick Answer:** conversational answers; searches when current evidence is needed.
- **Standard Research:** a concise, source-backed investigation.
- **Deep Research:** a more detailed report with optional lexical topic grouping.

## Workflow

The Streamlit app calls the shared Python research tools in-process:

```text
Streamlit UI -> question planning -> web search -> page extraction
             -> evidence analysis -> cited answer -> report exports
```

`server.py` exposes the same planning, search, extraction, clustering, and answer tools over FastMCP for local development. The deployed Streamlit app does not connect to an MCP endpoint.

## Evidence and citations

Answers cite retrieved sources using source IDs. Only citations that map to retrieved URLs are linked. The app distinguishes fetched page text from search-result snippets and notes when evidence is unavailable. Citation matching does not independently verify factual claims.

## Technology

Python, Streamlit, Google Gemini, Tavily, FastMCP, HTTPX, Beautiful Soup, and ReportLab. Optional topic grouping uses a small pure-Python TF-IDF/K-Means implementation.

## Project structure

```text
ResearchMate_AI/
|-- app.py                    # Streamlit interface
|-- config.py                 # Environment and Streamlit secrets
|-- list_gemini_models.py     # Local Gemini model diagnostic
|-- report_exports.py         # PDF report export
|-- research_workflow.py      # Shared workflow and report serialization
|-- researchmate.py           # Gemini planning, evidence, and citations
|-- server.py                 # Optional local FastMCP server
|-- requirements.txt
|-- tests/
|-- .env.example
`-- .github/workflows/tests.yml
```

## Installation

Use Python 3.11 for local development and deployment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and add your keys. The example file contains placeholders only.

## API configuration

Required for research:

```dotenv
GEMINI_API_KEY=your-gemini-api-key
TAVILY_API_KEY=your-tavily-api-key
```

Optional settings include `GEMINI_MODEL` and `TAVILY_SEARCH_DEPTH`. The local MCP server also accepts `MCP_HOST` and `MCP_PORT`. Configuration is read centrally from environment variables or `.env`; the Streamlit app also reads its configured Streamlit Secrets. Credential values are not displayed in the UI or included in report exports.

Never commit `.env` or Streamlit secrets. API usage is subject to provider quotas and limits.

## Run locally

From the project directory, start the Streamlit app:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed by Streamlit, usually <http://localhost:8501>. The app runs without a separate MCP server.

To run the optional MCP server for local MCP clients, open another terminal in the project directory:

```powershell
.\.venv\Scripts\python.exe server.py
```

The MCP server binds to `localhost` by default. Do not expose it publicly without authentication and appropriate network protections.

## Streamlit Community Cloud

The Streamlit deployment runs `app.py` and does not need a separate MCP server.

1. Push the reviewed project to GitHub.
2. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub.
3. Create a new app and select repository `Sanjuu887/ResearchMate_AI`.
4. Select branch `main` and entrypoint `app.py`.
5. In **Advanced settings**, choose Python **3.11**.
6. In the app's **Secrets** settings, add your own values:

   ```toml
   GEMINI_API_KEY = "your-gemini-api-key"
   GEMINI_MODEL = "your-enabled-gemini-model"
   TAVILY_API_KEY = "your-tavily-api-key"
   TAVILY_SEARCH_DEPTH = "advanced"
   ```

7. Deploy, then test all three modes and the Markdown, JSON, and PDF downloads.

Do not put API keys in GitHub files. Deployment has not been performed or verified by this project setup.

## Screenshots

Screenshots are not included yet. Add current application screenshots here after deployment.

## Tests

Run the offline suite from the repository root:

```powershell
.\.venv\Scripts\python.exe tests\run_feature_checks.py
```

Or use the standard-library test runner:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

These tests mock Gemini and Tavily. They validate application behavior but do not prove live credentials, provider availability, factual accuracy, or deployment success.
GitHub Actions runs this offline suite on pushes and pull requests to `main`.

To list Gemini model IDs available to a local API key:

```powershell
.\.venv\Scripts\python.exe list_gemini_models.py
```

The diagnostic prints model IDs only, not the API key.

## Known limitations

- Search quality and answer quality depend on external providers and the selected model.
- Search snippets and lightweight HTML extraction may omit relevant context.
- Citation mapping cannot guarantee that a source fully supports a claim.
- TF-IDF/K-Means groups word usage; it is not semantic analysis.
- A public Streamlit app can consume provider quota. Configure provider limits and monitor usage.

## License and attribution

This project builds on the existing ResearchMate AI repository and its FastMCP/Tavily workflow. Preserve original attribution. No separate `LICENSE` file is currently present; confirm the applicable license terms before redistribution.
