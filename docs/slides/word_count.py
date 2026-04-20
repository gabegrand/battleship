#!/usr/bin/env python3
"""Per-slide word count and speaking-time estimate for speaker_notes.md.

Run from this directory:

    python3 word_count.py

Prints a markdown table with one row per slide (in deck order), followed by
a totals row. Adjust `WPM` for a different speaking pace.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOTES_MD = HERE / "speaker_notes.md"

# Typical conference-talk pace is 150-160 wpm; midpoint.
WPM = 155


def parse_slides(text: str) -> list[tuple[str, str]]:
    """Return [(slide_id, body)] in deck order."""
    parts = re.split(r"^## #([^\n]+)$", text, flags=re.MULTILINE)
    slides: list[tuple[str, str]] = []
    for i in range(1, len(parts), 2):
        sid = parts[i].strip()
        # Cut at the next horizontal rule (or end of file).
        body = re.split(r"\n---\s*\n", parts[i + 1])[0]
        slides.append((sid, body))
    return slides


def count_words(body: str) -> int:
    """Count spoken words, stripping bullet markers and 'no notes' placeholder."""
    s = re.sub(r"^\s*-\s+", "", body, flags=re.MULTILINE)
    s = s.replace("*(No speaker notes.)*", "")
    return len(s.split())


def fmt_time(words: int) -> str:
    if words == 0:
        return "—"
    secs = round(words / WPM * 60)
    m, s = divmod(secs, 60)
    return f"{m}:{s:02d}"


def main() -> int:
    if not NOTES_MD.exists():
        print(
            f"error: {NOTES_MD} not found (run sync_speaker_notes.py first)",
            file=sys.stderr,
        )
        return 1

    slides = parse_slides(NOTES_MD.read_text())
    rows = [(sid, count_words(body)) for sid, body in slides]
    total_words = sum(w for _, w in rows)

    id_w = max(len(sid) + 1 for sid, _ in rows)  # +1 for the leading '#'
    id_w = max(id_w, len("**Total**"))
    wc_w = max(len("Words"), len(str(max((w for _, w in rows), default=0))))
    tm_w = max(len("Time"), max(len(fmt_time(w)) for _, w in rows))

    def row(label: str, words_str: str, time_str: str) -> str:
        return f"| {label.ljust(id_w)} | {words_str.rjust(wc_w)} | {time_str.rjust(tm_w)} |"

    sep = f"|{'-' * (id_w + 2)}|{'-' * (wc_w + 2)}|{'-' * (tm_w + 2)}|"

    print(row("Slide", "Words", "Time"))
    print(sep)
    for sid, w in rows:
        print(row(f"#{sid}", str(w), fmt_time(w)))
    print(sep)
    print(row("**Total**", str(total_words), fmt_time(total_words)))

    target = 10 * WPM  # 10-minute talk budget
    delta = total_words - target
    print(
        f"\n(@{WPM} wpm; 10:00 target = {target} words; "
        f"current is {abs(delta)} words {'over' if delta > 0 else 'under'})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
