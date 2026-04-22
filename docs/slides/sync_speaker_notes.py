#!/usr/bin/env python3
"""Regenerate speaker_notes.md from the <aside class="notes"> blocks in index.html.

Run from this directory:

    python3 sync_speaker_notes.py

Overwrites speaker_notes.md with a faithful markdown transcription of each
visible slide's speaker notes, in deck order. Sections with
`data-visibility="hidden"` are skipped, and sections without an <aside> are
flagged with "*(No speaker notes.)*".
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
INDEX_HTML = HERE / "index.html"
OUT_MD = HERE / "speaker_notes.md"


def extract_sections(html: str) -> list[tuple[str, str | None]]:
    """Return [(section_id, aside_inner_html_or_None)] for visible sections in deck order."""
    section_re = re.compile(r"<section\b([^>]*)>(.*?)</section>", re.DOTALL)
    results: list[tuple[str, str | None]] = []
    for m in section_re.finditer(html):
        attrs, body = m.group(1), m.group(2)
        if 'data-visibility="hidden"' in attrs:
            continue
        id_m = re.search(r'id="([^"]+)"', attrs)
        if not id_m:
            continue
        sid = id_m.group(1)
        aside_m = re.search(r'<aside class="notes">(.*?)</aside>', body, re.DOTALL)
        results.append((sid, aside_m.group(1) if aside_m else None))
    return results


def html_to_md(aside: str) -> str:
    """Convert an aside's inner HTML to markdown, preserving paragraph/list structure."""
    s = aside
    # Drop HTML comments outright.
    s = re.sub(r"<!--.*?-->", "", s, flags=re.DOTALL)
    # Inline tags.
    s = re.sub(r"<strong>(.*?)</strong>", r"**\1**", s, flags=re.DOTALL)
    s = re.sub(r"<em>(.*?)</em>", r"*\1*", s, flags=re.DOTALL)
    # Entities.
    replacements = {
        "&mdash;": "—",
        "&ndash;": "–",
        "&rsquo;": "'",
        "&lsquo;": "'",
        "&ldquo;": '"',
        "&rdquo;": '"',
        "&nbsp;": " ",
        "&amp;": "&",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)

    # Walk top-level <p>...</p> and <ul>...</ul> blocks in source order.
    blocks: list[str] = []
    token_re = re.compile(r"<p>(.*?)</p>|<ul>(.*?)</ul>", re.DOTALL)
    for m in token_re.finditer(s):
        p, ul = m.group(1), m.group(2)
        if p is not None:
            text = re.sub(r"\s+", " ", p).strip()
            if text:
                blocks.append(text)
        elif ul is not None:
            bullets: list[str] = []
            for item in re.findall(r"<li>(.*?)</li>", ul, flags=re.DOTALL):
                text = re.sub(r"\s+", " ", item).strip()
                if text:
                    bullets.append(f"- {text}")
            if bullets:
                blocks.append("\n".join(bullets))
    return "\n\n".join(blocks).strip()


def render_markdown(sections: list[tuple[str, str | None]]) -> str:
    lines: list[str] = [
        "# Speaker Notes",
        "",
        "Current speaker notes for each visible slide of the ICLR 2026 talk, "
        "extracted from `index.html`. Headings correspond to section ids — "
        "you can jump to a slide with `#/<id>` in the deck URL.",
        "",
    ]
    for sid, aside in sections:
        lines.append("---")
        lines.append("")
        lines.append(f"## #{sid}")
        lines.append("")
        lines.append(html_to_md(aside) if aside else "*(No speaker notes.)*")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    if not INDEX_HTML.exists():
        print(f"error: {INDEX_HTML} not found", file=sys.stderr)
        return 1
    sections = extract_sections(INDEX_HTML.read_text())
    OUT_MD.write_text(render_markdown(sections))
    print(f"Wrote {len(sections)} slides to {OUT_MD.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
