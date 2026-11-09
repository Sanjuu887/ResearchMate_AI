"""Complete research-report PDF export helpers."""

from __future__ import annotations

import html
import io
import re
import unicodedata
from xml.sax.saxutils import escape


def _pdf_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return normalized.encode("cp1252", "replace").decode("cp1252")


def _inline_markup(text: str) -> str:
    text = _pdf_text(text)
    links: list[str] = []

    def hold_link(match: re.Match) -> str:
        label = escape(match.group(1))
        url = html.escape(match.group(2), quote=True)
        links.append(f'<link href="{url}" color="#5146c7">{label}</link>')
        return f"RM_LINK_{len(links) - 1}_TOKEN"

    text = re.sub(
        r"\[([^\]]+)\]\(\s*<?(https?://[^)\s>]+)>?\s*\)",
        hold_link,
        text,
    )
    text = escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', text)
    for index, link in enumerate(links):
        text = text.replace(f"RM_LINK_{index}_TOKEN", link)
    return text


def _markdown_flowables(markdown_text: str, styles: dict, usable_width: float) -> list:
    from reportlab.lib import colors
    from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

    lines = _pdf_text(markdown_text).splitlines()
    flowables = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            flowables.append(Spacer(1, 5))
            index += 1
            continue

        if line.startswith("|") and index + 1 < len(lines):
            separator = lines[index + 1].strip()
            if separator.startswith("|") and re.fullmatch(r"[\s|:-]+", separator):
                table_lines = [line]
                index += 2
                while index < len(lines) and lines[index].strip().startswith("|"):
                    table_lines.append(lines[index].strip())
                    index += 1
                data = [
                    [Paragraph(_inline_markup(cell), styles["table"]) for cell in _table_cells(row)]
                    for row in table_lines
                ]
                column_count = max((len(row) for row in data), default=1)
                widths = [usable_width / column_count] * column_count
                table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
                table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef0fb")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#18243a")),
                            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dfe4ee")),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 6),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                            ("TOPPADDING", (0, 0), (-1, -1), 5),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                        ]
                    )
                )
                flowables.extend([table, Spacer(1, 9)])
                continue

        heading = re.match(r"^(#{1,4})\s+(.+)$", line)
        if heading:
            level = min(len(heading.group(1)), 3)
            flowables.append(Paragraph(_inline_markup(heading.group(2)), styles[f"h{level}"]))
        elif re.match(r"^[-*]\s+", line):
            body = re.sub(r"^[-*]\s+", "", line)
            flowables.append(Paragraph(f"&bull;&nbsp; {_inline_markup(body)}", styles["bullet"]))
        elif re.match(r"^\d+[.)]\s+", line):
            numbered = re.match(r"^(\d+)[.)]\s+(.*)$", line)
            flowables.append(
                Paragraph(
                    f"{numbered.group(1)}.&nbsp; {_inline_markup(numbered.group(2))}",
                    styles["bullet"],
                )
            )
        elif line.startswith(">"):
            flowables.append(Paragraph(_inline_markup(line[1:].strip()), styles["quote"]))
        else:
            flowables.append(Paragraph(_inline_markup(line), styles["body"]))
        index += 1
    return flowables


def _table_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def make_report_pdf(question: str, result: dict) -> bytes:
    """Create a paginated PDF containing the complete answer and its evidence."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    buffer = io.BytesIO()
    page_width, _ = A4
    left_margin = right_margin = 19 * mm
    usable_width = page_width - left_margin - right_margin
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=right_margin,
        leftMargin=left_margin,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="ResearchMate AI — Research Report",
        author="ResearchMate AI",
    )
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "RMTitle", parent=base["Title"], textColor=colors.HexColor("#18243a"),
            fontName="Helvetica-Bold", fontSize=23, leading=28, alignment=TA_LEFT,
            spaceAfter=8,
        ),
        "question": ParagraphStyle(
            "RMQuestion", parent=base["Normal"], textColor=colors.HexColor("#36445a"),
            fontSize=12, leading=17, spaceAfter=9,
        ),
        "meta": ParagraphStyle(
            "RMMeta", parent=base["Normal"], textColor=colors.HexColor("#68758a"),
            fontSize=9, leading=13, spaceAfter=13,
        ),
        "h1": ParagraphStyle(
            "RMH1", parent=base["Heading1"], textColor=colors.HexColor("#18243a"),
            fontSize=17, leading=21, spaceBefore=13, spaceAfter=7, keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "RMH2", parent=base["Heading2"], textColor=colors.HexColor("#273650"),
            fontSize=14, leading=18, spaceBefore=11, spaceAfter=6, keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "RMH3", parent=base["Heading3"], textColor=colors.HexColor("#36445a"),
            fontSize=11.5, leading=15, spaceBefore=9, spaceAfter=5, keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "RMBody", parent=base["BodyText"], textColor=colors.HexColor("#28364b"),
            fontName="Helvetica", fontSize=9.6, leading=14.2, spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "RMBullet", parent=base["BodyText"], textColor=colors.HexColor("#28364b"),
            fontSize=9.6, leading=14, leftIndent=11, firstLineIndent=-2, spaceAfter=4,
        ),
        "quote": ParagraphStyle(
            "RMQuote", parent=base["BodyText"], textColor=colors.HexColor("#53627a"),
            fontSize=9.3, leading=13, leftIndent=12, borderColor=colors.HexColor("#b9b4ef"),
            borderWidth=2, borderPadding=6, spaceBefore=4, spaceAfter=8,
        ),
        "table": ParagraphStyle(
            "RMTable", parent=base["BodyText"], textColor=colors.HexColor("#28364b"),
            fontSize=8, leading=10.5,
        ),
        "source": ParagraphStyle(
            "RMSource", parent=base["BodyText"], textColor=colors.HexColor("#28364b"),
            fontSize=9, leading=12.5, spaceAfter=4,
        ),
        "excerpt": ParagraphStyle(
            "RMExcerpt", parent=base["BodyText"], textColor=colors.HexColor("#53627a"),
            fontSize=8.5, leading=12, leftIndent=8, spaceAfter=9,
        ),
    }

    story = [
        Paragraph("ResearchMate AI", styles["title"]),
        Paragraph(_inline_markup(question), styles["question"]),
    ]
    generated_at = result.get("generated_at", "")
    depth = result.get("depth", "")
    story.append(
        Paragraph(
            _inline_markup(f"Research depth: {depth}  ·  Generated: {generated_at}"),
            styles["meta"],
        )
    )
    if result.get("note"):
        story.append(
            Paragraph(
                f"<b>Research note:</b> {_inline_markup(str(result['note']))}",
                styles["quote"],
            )
        )

    story.extend(_markdown_flowables(result.get("answer", ""), styles, usable_width))
    sources = result.get("sources", [])
    if sources:
        story.extend(
            [
                Spacer(1, 10),
                Paragraph("Evidence &amp; References", styles["h1"]),
            ]
        )
        for source in sources:
            try:
                citation_number = int(str(source.get("id", "")).removeprefix("S"))
            except ValueError:
                continue
            title = escape(_pdf_text(str(source.get("title") or source.get("url") or "Source")))
            story.append(
                Paragraph(f"<b>[{citation_number}] {title}</b>", styles["source"])
            )
            publisher = str(source.get("publisher") or "").strip()
            published = str(source.get("published_date") or "").strip()
            retrieved_as = source.get("retrieved_as", "search snippet")
            details = [
                "Fetched page content" if retrieved_as == "page" else "Search snippet only"
            ]
            if publisher:
                details.append(f"Publisher: {_pdf_text(publisher)}")
            if published:
                details.append(f"Published: {_pdf_text(published)}")
            story.append(
                Paragraph(_inline_markup(" · ".join(details)), styles["meta"])
            )
            url = str(source.get("url") or "").strip()
            if url.startswith(("https://", "http://")):
                safe_url = html.escape(url, quote=True)
                story.append(
                    Paragraph(
                        f'<link href="{safe_url}" color="#5146c7">{escape(_pdf_text(url))}</link>',
                        styles["source"],
                    )
                )
            excerpt = str(source.get("content") or "").strip()
            if excerpt:
                story.append(
                    Paragraph(
                        _inline_markup(excerpt[:1800]),
                        styles["excerpt"],
                    )
                )
            story.append(Spacer(1, 4))

    def draw_page(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#e3e8f0"))
        canvas.line(left_margin, 13 * mm, page_width - right_margin, 13 * mm)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#68758a"))
        canvas.drawString(left_margin, 9 * mm, "ResearchMate AI · Verify important claims against sources")
        canvas.drawRightString(page_width - right_margin, 9 * mm, str(document.page))
        canvas.restoreState()

    doc.build(story, onFirstPage=draw_page, onLaterPages=draw_page)
    return buffer.getvalue()
