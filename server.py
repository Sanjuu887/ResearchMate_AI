"""
ResearchMate AI — FastMCP Server
=====================================
Research planning, search, extraction, clustering, and answer-generation tools.

Run with:  python server.py
"""

import re
import json
import math
import asyncio
import hashlib
from typing import Any
from datetime import datetime, timezone
from collections import defaultdict, Counter
from urllib.parse import urlsplit

import httpx
from bs4 import BeautifulSoup
from fastmcp import FastMCP
from config import get_config
from researchmate import ResearchError, plan_question, synthesize_answer

# ── Init ──────────────────────────────────────────────────────────────────────
mcp = FastMCP(
    name="researchmate-ai",
    instructions="""
    ResearchMate AI uses plan_research to decide whether evidence is needed.
    For research, use focused search_web queries, fetch_and_chunk for selected
    pages, and generate_answer with source IDs and evidence. Clustering is
    optional lexical TF-IDF grouping, not semantic analysis.
    """,
)

MAX_PAGE_BYTES = 1_500_000


@mcp.tool()
async def plan_research(
    question: str,
    depth: str = "standard",
    conversation_history: list[dict] | None = None,
) -> dict[str, Any]:
    """Decide whether to search and return focused queries for a question."""
    try:
        plan = await plan_question(question, depth, conversation_history)
        return {"success": True, **plan}
    except (ResearchError, ValueError) as exc:
        return _error(str(exc))


@mcp.tool()
async def generate_answer(
    question: str,
    sources: list[dict],
    depth: str = "standard",
    needs_research: bool = True,
    conversation_history: list[dict] | None = None,
) -> dict[str, Any]:
    """Generate a direct answer or a source-grounded answer when research is needed."""
    try:
        answer = await synthesize_answer(
            question, sources, depth, needs_research, conversation_history
        )
        return {"success": True, "answer": answer}
    except (ResearchError, ValueError) as exc:
        return _error(str(exc))


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 1 — search_web
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def search_web(
    query: str,
    max_results: int = 8,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
) -> dict[str, Any]:
    """
    Search the web using Tavily API.

    Args:
        query:           The search query string
        max_results:     Number of results to return (default: 8, max: 15)
        include_domains: Optional list of domains to prioritize (e.g. [".edu", ".gov"])
        exclude_domains: Optional list of domains to exclude

    Returns:
        dict with keys:
          - query: the original query
          - results: list of {title, url, snippet, score, published_date}
          - total_found: count of results
          - search_id: unique ID for this search session
    """
    query = query.strip()
    if not query:
        return _error("Search query cannot be empty.")
    if len(query) > 400:
        return _error("Search query must be 400 characters or fewer.")
    config = get_config()
    if not config.tavily_api_key:
        return _error(
            "TAVILY_API_KEY is missing. Set it in your local .env file or the "
            "Streamlit app's Secrets settings."
        )

    max_results = min(max(1, max_results), 15)

    payload: dict[str, Any] = {
        "query": query,
        "search_depth": config.tavily_search_depth,
        "max_results": max_results,
        "include_answer": True,
        "include_raw_content": False,
    }
    if include_domains:
        payload["include_domains"] = include_domains
    if exclude_domains:
        payload["exclude_domains"] = exclude_domains

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                headers={"Authorization": f"Bearer {config.tavily_api_key}"},
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
        if not isinstance(data, dict) or not isinstance(data.get("results", []), list):
            return _error("Tavily returned a malformed search response.")

        results = []
        for r in data.get("results", []):
            if not isinstance(r, dict):
                continue
            url = r.get("url", "")
            parsed_url = urlsplit(url) if isinstance(url, str) else None
            if (
                parsed_url is None
                or parsed_url.scheme not in {"http", "https"}
                or not parsed_url.hostname
            ):
                continue
            try:
                score = float(r.get("score", 0.0) or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            if not math.isfinite(score):
                score = 0.0
            results.append(
                {
                    "title": str(r.get("title", ""))[:300],
                    "url": url,
                    "snippet": str(r.get("content", ""))[:500],
                    "score": round(score, 3),
                    "published_date": str(r.get("published_date", ""))[:100],
                    "publisher": str(r.get("publisher") or r.get("source") or "")[:200],
                }
            )

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)

        return {
            "query": query,
            "answer_summary": data.get("answer", ""),
            "results": results,
            "total_found": len(results),
            "search_id": _short_hash(query),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            return _error(
                "Tavily rejected the configured API key. Verify that "
                "TAVILY_API_KEY is active and correct in your local .env file "
                "or Streamlit Secrets, then restart the app."
            )
        return _error(f"Tavily returned HTTP {e.response.status_code}.")
    except httpx.TimeoutException:
        return _error("Tavily search timed out. Please try again.")
    except httpx.RequestError as e:
        return _error(f"Could not reach Tavily: {e}.")
    except (ValueError, TypeError):
        return _error("Tavily returned a malformed response.")


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 2 — fetch_and_chunk
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def fetch_and_chunk(
    urls: list[str],
    chunk_size: int = 400,
    chunk_overlap: int = 50,
    max_chunks_per_url: int = 6,
) -> dict[str, Any]:
    """
    Fetch web pages and split their content into overlapping text chunks.

    Args:
        urls:               List of URLs to fetch (max 10)
        chunk_size:         Target word count per chunk (default: 400)
        chunk_overlap:      Word overlap between consecutive chunks (default: 50)
        max_chunks_per_url: Max chunks to keep per URL (default: 6)

    Returns:
        dict with keys:
          - chunks: list of {id, url, title, text, word_count, chunk_index}
          - total_chunks: total number of chunks
          - fetched_urls: count of successfully fetched URLs
          - failed_urls: list of URLs that failed
    """
    urls = list(dict.fromkeys(url for url in urls if isinstance(url, str)))[:10]
    chunk_size = min(max(chunk_size, 100), 800)
    chunk_overlap = min(max(chunk_overlap, 0), chunk_size - 1)
    max_chunks_per_url = min(max(max_chunks_per_url, 1), 8)

    chunks_all: list[dict] = []
    failed: list[str] = []
    failed_pages: list[dict[str, str]] = []

    async with httpx.AsyncClient(
        timeout=20,
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"},
    ) as client:
        tasks = [_fetch_one(client, url) for url in urls]
        pages = await asyncio.gather(*tasks, return_exceptions=True)

    for url, page in zip(urls, pages):
        if isinstance(page, Exception) or page is None:
            failed.append(url)
            failed_pages.append({"url": url, "error": str(page or "Fetch failed")})
            continue
        if "error" in page:
            failed.append(url)
            failed_pages.append({"url": url, "error": page["error"]})
            continue

        raw_chunks = _chunk_text(
            page["text"], chunk_size, chunk_overlap, max_chunks_per_url
        )
        for i, text in enumerate(raw_chunks):
            chunk_id = f"{_short_hash(url)}-{i}"
            chunks_all.append(
                {
                    "id": chunk_id,
                    "url": url,
                    "title": page.get("title", ""),
                    "text": text,
                    "word_count": len(text.split()),
                    "chunk_index": i,
                }
            )

    return {
        "chunks": chunks_all,
        "total_chunks": len(chunks_all),
        "fetched_urls": len(urls) - len(failed),
        "failed_urls": failed,
        "failed_pages": failed_pages,
    }


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 3 — cluster_findings
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def cluster_findings(
    chunks: list[dict],
    n_clusters: int = 4,
    top_terms_per_cluster: int = 8,
) -> dict[str, Any]:
    """
    Group text chunks by lexical similarity using TF-IDF + K-Means.

    Args:
        chunks:                List of chunk dicts (from fetch_and_chunk output)
        n_clusters:            Number of thematic clusters to create (default: 4)
        top_terms_per_cluster: Number of top keywords to surface per cluster (default: 8)

    Returns:
        dict with keys:
          - clusters: list of {cluster_id, label, keywords, chunks, summary}
          - n_clusters: actual cluster count used
          - method: algorithm description
    """
    if not chunks:
        return _error("No chunks provided to cluster.")

    texts = [c.get("text", "") for c in chunks]
    n_clusters = min(max(n_clusters, 1), len(texts))

    try:
        # Pure stdlib TF-IDF (no scikit-learn dependency needed)
        tfidf_matrix, vocab = _tfidf(texts)
        labels = _kmeans(tfidf_matrix, n_clusters, max_iter=100)
        top_terms = _top_terms_per_cluster(tfidf_matrix, labels, vocab, top_terms_per_cluster)

        # Group chunks by cluster
        clustered: dict[int, list] = defaultdict(list)
        for chunk, label in zip(chunks, labels):
            clustered[label].append(chunk)

        clusters_out = []
        for cid in sorted(clustered.keys()):
            cluster_chunks = clustered[cid]
            keywords = top_terms.get(cid, [])
            label = _infer_cluster_label(keywords)
            summary = _cluster_summary(cluster_chunks)

            clusters_out.append(
                {
                    "cluster_id": cid,
                    "label": label,
                    "keywords": keywords,
                    "chunk_count": len(cluster_chunks),
                    "chunks": cluster_chunks,
                    "summary": summary,
                    "source_urls": list({c["url"] for c in cluster_chunks}),
                }
            )

        return {
            "clusters": clusters_out,
            "n_clusters": len(clusters_out),
            "total_chunks_clustered": len(chunks),
            "method": "TF-IDF vectorization + K-Means clustering (pure Python)",
        }

    except Exception as e:
        return _error(f"Clustering failed: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 4 — generate_report
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def generate_report(
    topic: str,
    clusters: list[dict],
    format: str = "markdown",
    include_sources: bool = True,
) -> dict[str, Any]:
    """
    Synthesize clustered research findings into a structured executive report.

    Args:
        topic:           The research topic / title
        clusters:        List of cluster dicts (from cluster_findings output)
        format:          Output format — "markdown" or "json" (default: "markdown")
        include_sources: Whether to append a sources section (default: True)

    Returns:
        dict with keys:
          - report: formatted report string (markdown or JSON)
          - topic: original topic
          - sections: count of sections generated
          - word_count: approximate word count
          - generated_at: ISO timestamp
    """
    if not clusters:
        return _error("No clusters provided to generate report from.")

    # ── Collect all unique sources ──
    all_sources: dict[str, str] = {}
    for cluster in clusters:
        for chunk in cluster.get("chunks", []):
            url = chunk.get("url", "")
            title = chunk.get("title", url)
            if url:
                all_sources[url] = title

    # ── Build report ──
    if format not in {"markdown", "json"}:
        return _error("Report format must be 'markdown' or 'json'.")

    now = datetime.now(timezone.utc).strftime("%B %d, %Y · %H:%M UTC")
    lines: list[str] = []

    if format == "markdown":
        lines += [
            f"# ResearchMate AI — Research Report",
            f"## {topic}",
            f"",
            f"> **Generated:** {now}  ",
            f"> **Sources analyzed:** {len(all_sources)}  ",
            f"> **Thematic clusters:** {len(clusters)}  ",
            f"> **Method:** TF-IDF lexical clustering with extractive summaries",
            f"",
            "---",
            "",
            "## Executive Summary",
            "",
            _executive_summary(topic, clusters),
            "",
            "---",
            "",
        ]

        for i, cluster in enumerate(clusters, 1):
            label = cluster.get("label", f"Theme {i}")
            keywords = cluster.get("keywords", [])
            summary = cluster.get("summary", "")
            chunk_count = cluster.get("chunk_count", 0)
            source_urls = cluster.get("source_urls", [])

            lines += [
                f"## Cluster {i}: {label}",
                f"",
                f"**Keywords:** {', '.join(keywords)}  ",
                f"**Sources in cluster:** {len(source_urls)}  ",
                f"**Text chunks analyzed:** {chunk_count}",
                f"",
                summary,
                "",
            ]

            # Sample supporting excerpts (top 2 chunks)
            top_chunks = cluster.get("chunks", [])[:2]
            if top_chunks:
                lines.append("**Supporting evidence:**")
                lines.append("")
                for chunk in top_chunks:
                    excerpt = chunk.get("text", "")[:300].replace("\n", " ").strip()
                    source_title = chunk.get("title", chunk.get("url", ""))[:60]
                    lines += [
                        f"> {excerpt}…",
                        f">",
                        f"> *— {source_title}*",
                        f"",
                    ]

            lines.append("---")
            lines.append("")

        # Key findings section
        lines += [
            "## Key Findings",
            "",
        ]
        for i, cluster in enumerate(clusters, 1):
            kw = cluster.get("keywords", [""])
            lines.append(f"- **{cluster.get('label', f'Theme {i}')}**: {', '.join(kw[:4])}")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Sources
        if include_sources and all_sources:
            lines += ["## Sources", ""]
            for j, (url, title) in enumerate(all_sources.items(), 1):
                display = title[:80] if title != url else url[:80]
                lines.append(f"{j}. [{display}]({url})")
            lines.append("")

        lines += [
            "---",
            f"*Report generated by ResearchMate AI · {now}*",
        ]

    elif format == "json":
        report_data = {
            "topic": topic,
            "generated_at": now,
            "sources_count": len(all_sources),
            "executive_summary": _executive_summary(topic, clusters),
            "clusters": [
                {
                    "id": c.get("cluster_id"),
                    "label": c.get("label"),
                    "keywords": c.get("keywords"),
                    "summary": c.get("summary"),
                    "source_count": len(c.get("source_urls", [])),
                }
                for c in clusters
            ],
            "sources": [{"url": u, "title": t} for u, t in all_sources.items()],
        }
        report_str = json.dumps(report_data, indent=2)
        return {
            "report": report_str,
            "topic": topic,
            "sections": len(clusters),
            "word_count": len(report_str.split()),
            "generated_at": now,
        }

    report_str = "\n".join(lines)
    return {
        "report": report_str,
        "topic": topic,
        "sections": len(clusters) + 2,  # clusters + summary + findings
        "word_count": len(report_str.split()),
        "generated_at": now,
    }


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _error(msg: str) -> dict:
    return {"error": msg, "success": False}


def _short_hash(s: str, length: int = 8) -> str:
    return hashlib.md5(s.encode()).hexdigest()[:length]


async def _fetch_one(client: httpx.AsyncClient, url: str) -> dict | None:
    """Fetch a single URL and extract clean text."""
    try:
        parsed = urlsplit(url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
        ):
            return {"error": "Only HTTP(S) URLs without embedded credentials are accepted."}
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            content_length = response.headers.get("content-length")
            if content_length and int(content_length) > MAX_PAGE_BYTES:
                return {"error": "Page exceeds the 1.5 MB content limit."}
            body = bytearray()
            async for part in response.aiter_bytes():
                body.extend(part)
                if len(body) > MAX_PAGE_BYTES:
                    return {"error": "Page exceeds the 1.5 MB content limit."}
            html = bytes(body).decode(response.encoding or "utf-8", errors="replace")

        title, text = _extract_page_text(html, url)
        if len(text) < 100:
            return {"error": "Page too short"}

        return {"url": url, "title": _clean_text(title), "text": text}

    except httpx.HTTPStatusError as e:
        return {"error": f"Page returned HTTP {e.response.status_code}."}
    except httpx.TimeoutException:
        return {"error": "Page fetch timed out."}
    except (httpx.RequestError, UnicodeError, ValueError) as e:
        return {"error": f"Could not fetch or decode page: {e}."}


def _extract_page_text(html: str, url: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else url
    for element in soup(
        [
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "aside",
            "svg",
            "figure",
            "form",
            "noscript",
            "iframe",
        ]
    ):
        element.decompose()
    content_root = soup.find(["main", "article"]) or soup.body or soup
    text = re.sub(r"\s+", " ", content_root.get_text(" ", strip=True)).strip()
    return _clean_text(title), text[:60_000]


def _clean_text(t: str) -> str:
    import html as html_mod
    t = re.sub(r"<[^>]+>", "", t)
    t = html_mod.unescape(t)
    return re.sub(r"\s+", " ", t).strip()


def _chunk_text(
    text: str,
    chunk_size: int,
    overlap: int,
    max_chunks: int,
) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    if not words:
        return []

    step = max(1, chunk_size - overlap)
    chunks = []
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size])
        if len(chunk.split()) >= 50:  # Skip tiny trailing chunks
            chunks.append(chunk)
        if len(chunks) >= max_chunks:
            break

    return chunks


# ── Pure-Python TF-IDF ────────────────────────────────────────────────────────

STOPWORDS = {
    # Articles / conjunctions / prepositions
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "by","from","as","not","no","so","if","into","over","out","up","than",
    # Verbs / auxiliaries
    "is","are","was","were","be","been","being","have","has","had",
    "do","does","did","will","would","could","should","may","might","can",
    # Pronouns / determiners
    "this","that","these","those","it","its","we","they","he","she",
    "you","i","our","their","your","my","all","also","more","about",
    # HTML entity remnants that slip through after unescape
    "nbsp","rsquo","ldquo","rdquo","mdash","ndash","amp","quot","apos",
    "lsquo","hellip","copy","reg","trade","euro","usd","eur",
    # Web/UI boilerplate noise
    "skip","content","menu","home","contact","get","demo","cookie","cookies",
    "privacy","terms","login","sign","subscribe","share","follow","click",
    "read","learn","see","back","next","prev","page","view","load",
    # Generic filler words
    "new","one","two","use","used","uses","using","make","made","well",
    "just","even","such","each","both","many","most","some","any","via",
    "per","yet","now","how","why","who","what","when","where","while",
    "then","thus","however","therefore","including","based","within",
    # Single-word brand/geo noise that poisons keywords
    "america","american","bank","banks","global","world","international",
    "product","products","production","service","services","company",
    "companies","business","businesses","industry","industries","sector",
    "icl","cio","llc","inc","ltd","corp","plc","gmbh","pty",
    "january","february","march","april","june","july","august",
    "september","october","november","december","mon","tue","wed",
    "tech","technology","technologies","digital","online","platform",
    "report","reports","article","articles","blog","news","press",
    "said","says","according","noted","stated","told","announced",
    "trend","trends","trending","life","end","start","began","began",
}

def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z]{3,}", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def _tfidf(docs: list[str]) -> tuple[list[list[float]], list[str]]:
    """Returns (tfidf_matrix, vocab_list)."""
    tokenized = [_tokenize(d) for d in docs]
    N = len(tokenized)

    # Build vocabulary (top 500 terms by doc freq)
    df: Counter = Counter()
    for tokens in tokenized:
        df.update(set(tokens))

    vocab = [term for term, _ in df.most_common(500) if df[term] > 1]
    vocab_idx = {t: i for i, t in enumerate(vocab)}
    V = len(vocab)

    matrix: list[list[float]] = []
    for tokens in tokenized:
        tf: Counter = Counter(tokens)
        total = max(len(tokens), 1)
        row = [0.0] * V
        for term, count in tf.items():
            if term in vocab_idx:
                tf_val = count / total
                idf_val = math.log((N + 1) / (df[term] + 1)) + 1
                row[vocab_idx[term]] = tf_val * idf_val
        # L2-normalize
        norm = math.sqrt(sum(x * x for x in row)) or 1.0
        matrix.append([x / norm for x in row])

    return matrix, vocab


def _kmeans(
    matrix: list[list[float]],
    k: int,
    max_iter: int = 100,
) -> list[int]:
    """Simple K-Means clustering on a dense float matrix."""
    import random
    random.seed(42)
    n, d = len(matrix), len(matrix[0]) if matrix else 0
    k = min(k, n)

    # Init centroids — K-Means++ style (pick spread-out seeds)
    centroids = [matrix[random.randint(0, n - 1)]]
    for _ in range(k - 1):
        dists = [min(_cosine_dist(p, c) for c in centroids) for p in matrix]
        total = sum(dists) or 1
        probs = [d / total for d in dists]
        cumulative = 0
        r = random.random()
        chosen = n - 1
        for idx, p in enumerate(probs):
            cumulative += p
            if cumulative >= r:
                chosen = idx
                break
        centroids.append(matrix[chosen])

    labels = [0] * n
    for _ in range(max_iter):
        # Assignment
        new_labels = [
            min(range(k), key=lambda ci: _cosine_dist(matrix[i], centroids[ci]))
            for i in range(n)
        ]
        if new_labels == labels:
            break
        labels = new_labels

        # Update centroids
        for ci in range(k):
            cluster_pts = [matrix[i] for i, l in enumerate(labels) if l == ci]
            if cluster_pts:
                centroids[ci] = [
                    sum(p[j] for p in cluster_pts) / len(cluster_pts)
                    for j in range(d)
                ]

    return labels


def _cosine_dist(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return 1.0 - dot  # vectors are already normalized


def _top_terms_per_cluster(
    matrix: list[list[float]],
    labels: list[int],
    vocab: list[str],
    top_n: int,
) -> dict[int, list[str]]:
    """For each cluster, average TF-IDF scores and return top terms."""
    cluster_sums: dict[int, list[float]] = defaultdict(lambda: [0.0] * len(vocab))
    cluster_counts: Counter = Counter()

    for vec, label in zip(matrix, labels):
        sums = cluster_sums[label]
        for i, v in enumerate(vec):
            sums[i] += v
        cluster_counts[label] += 1

    result = {}
    for cid, sums in cluster_sums.items():
        count = cluster_counts[cid]
        avg = [s / count for s in sums]
        top_indices = sorted(range(len(avg)), key=lambda i: avg[i], reverse=True)[:top_n]
        result[cid] = [vocab[i] for i in top_indices]

    return result


def _infer_cluster_label(keywords: list[str]) -> str:
    """Map top keywords to a human-readable cluster label."""
    kw_set = set(keywords[:8])

    LABEL_MAP = [
        # Energy & power
        ({"energy","power","electricity","grid","consumption","demand","twh","kwh","megawatt","gigawatt","load","utility"}, "Energy Demand & Grid"),
        # Water
        ({"water","cooling","thermal","liquid","immersion","heat","evaporation","drought","reservoir","gallons","cubic"}, "Cooling & Water Use"),
        # Climate / renewables
        ({"nuclear","renewable","solar","wind","carbon","emissions","clean","green","climate","sustainability","net","zero"}, "Clean Energy & Sustainability"),
        # Infrastructure
        ({"data","center","infrastructure","cloud","server","hardware","rack","facility","hyperscale","campus"}, "Data Center Infrastructure"),
        # AI / compute
        ({"ai","model","training","inference","llm","gpu","compute","nvidia","chip","accelerator","transformer","language"}, "AI Compute & Models"),
        # Quantum
        ({"quantum","qubit","qubits","superposition","entanglement","decoherence","fault","logical","error","correction","ibm","google"}, "Quantum Computing"),
        # Biotech / health
        ({"drug","protein","gene","genome","clinical","trial","cancer","therapy","mrna","vaccine","biotech","pharma","disease"}, "Biotech & Health"),
        # Policy / regulation
        ({"policy","regulation","government","law","standard","congress","eu","act","bill","rule","compliance","regulatory"}, "Policy & Regulation"),
        # Market / finance
        ({"market","billion","million","investment","revenue","funding","startup","ipo","valuation","growth","percent","forecast"}, "Market & Economic Trends"),
        # Security / cyber
        ({"security","cyber","attack","breach","vulnerability","encryption","threat","malware","ransomware","phishing"}, "Cybersecurity"),
        # Space
        ({"space","satellite","rocket","orbit","nasa","spacex","lunar","mars","launch","payload","constellation"}, "Space Technology"),
        # Robotics / automation
        ({"robot","autonomous","automation","drone","vehicle","manufacturing","warehouse","arm","sensor","lidar"}, "Robotics & Automation"),
        # Research / science
        ({"research","study","paper","journal","university","laboratory","experiment","discovery","breakthrough","findings"}, "Research & Science"),
    ]

    best_label, best_score = "Emerging Trends", 0
    for key_terms, label in LABEL_MAP:
        score = len(kw_set & key_terms)
        if score > best_score:
            best_score, best_label = score, label

    # Fallback: build label from top 2 meaningful keywords
    if best_score == 0 and keywords:
        clean = [k for k in keywords if len(k) > 4][:2]
        if clean:
            return " & ".join(w.title() for w in clean)

    return best_label


def _cluster_summary(chunks: list[dict]) -> str:
    """Create a brief summary paragraph from a cluster's chunks."""
    if not chunks:
        return "No content available for this cluster."

    # Pick the most representative chunks (first 3)
    sample = chunks[:3]
    combined = " ".join(c.get("text", "")[:200] for c in sample)

    # Extract the first ~2 sentences worth of content
    sentences = re.split(r"(?<=[.!?])\s+", combined)
    summary_sentences = [s.strip() for s in sentences if len(s.split()) > 8][:3]

    if not summary_sentences:
        return combined[:400]

    return " ".join(summary_sentences)


def _executive_summary(topic: str, clusters: list[dict]) -> str:
    """Generate a brief executive summary from cluster labels and keywords."""
    themes = [c.get("label", f"Theme {i+1}") for i, c in enumerate(clusters)]
    kws = []
    for c in clusters:
        kws.extend(c.get("keywords", [])[:2])
    unique_kws = list(dict.fromkeys(kws))[:8]

    return (
        f"This report synthesizes research on **{topic}** across "
        f"{len(clusters)} thematic clusters: {', '.join(themes)}. "
        f"Key concepts emerging from the analysis include: "
        f"{', '.join(unique_kws)}. "
        f"The following sections present findings from each cluster with "
        f"supporting evidence drawn from primary sources."
    )


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    config = get_config()
    host = config.mcp_host
    port = config.mcp_port

    print("ResearchMate AI MCP Server")
    print(f"   Host : {host}")
    print(f"   Port : {port}")
    print("   Tools: plan_research, search_web, fetch_and_chunk,")
    print("          cluster_findings, generate_answer, generate_report")
    print(
        "   Tavily key: configured"
        if config.tavily_api_key
        else "   Tavily key: missing (set TAVILY_API_KEY in .env)"
    )
    print(
        "   Gemini key: configured"
        if config.gemini_api_key
        else "   Gemini key: missing (set GEMINI_API_KEY in .env)"
    )
    print()

    mcp.run(transport="streamable-http", host=host, port=port)