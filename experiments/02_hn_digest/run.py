"""
Experiment 02 — HN Digest

A pipeline of 1-bit agents processes Hacker News: fetches top stories,
summarizes each article in sequence, and a curator produces a ranked
digest.

Pipeline: fetch stories → fetch content → summarize each → curate digest

Each summarizer and the curator are separate agent invocations with
distinct prompts — like a team of specialist micro-agents on a
production line.

Usage:
  python -m experiments.02_hn_digest.run
  python -m experiments.02_hn_digest.run --stories 10
  python -m experiments.02_hn_digest.run --stories 3 --model 1.7b
"""

import argparse
import html
import re
import textwrap
import time
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx
from mlx_lm.sample_utils import make_sampler
from rich.panel import Panel
from rich.table import Table

from shared.console import (
    banner,
    console,
    print_turn,
    section,
    stats_footer,
)
from shared.models import MODELS, load_model

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HN_API = "https://hacker-news.firebaseio.com/v0"

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class Story:
    id: int
    title: str
    url: str | None
    score: int
    comments: int
    text: str | None


@dataclass
class StoryWithSummary:
    story: Story
    content_len: int
    summary: str
    elapsed_s: float


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def fetch_top_stories(count: int = 5) -> list[Story]:
    with httpx.Client(timeout=15) as client:
        ids = client.get(f"{HN_API}/topstories.json").json()[:count * 2]
        stories = []
        for sid in ids:
            if len(stories) >= count:
                break
            item = client.get(f"{HN_API}/item/{sid}.json").json()
            if item and item.get("type") == "story" and not item.get("dead"):
                stories.append(Story(
                    id=item["id"],
                    title=item.get("title", ""),
                    url=item.get("url"),
                    score=item.get("score", 0),
                    comments=item.get("descendants", 0),
                    text=item.get("text"),
                ))
        return stories


def fetch_article_text(url: str, max_chars: int = 3000) -> str | None:
    try:
        with httpx.Client(
            timeout=10,
            follow_redirects=True,
            headers={"User-Agent": "miniagents/0.1"},
        ) as client:
            resp = client.get(url)
            resp.raise_for_status()
            if "text/html" not in resp.headers.get("content-type", ""):
                return None
        text = resp.text
        text = re.sub(r"<(script|style|nav|footer|header)[^>]*>.*?</\1>", "", text, flags=re.S | re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        text = html.unescape(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_chars] if len(text) > 100 else None
    except Exception:
        return None


def get_story_content(story: Story, max_chars: int = 3000) -> str:
    if story.text:
        raw = html.unescape(re.sub(r"<[^>]+>", " ", story.text))
        return raw[:max_chars]
    if story.url:
        content = fetch_article_text(story.url, max_chars)
        if content:
            return content
    return story.title


# ---------------------------------------------------------------------------
# Agent prompts
# ---------------------------------------------------------------------------

SUMMARIZER_SYSTEM = (
    "You summarize articles in 2-3 sentences. "
    "Focus on what happened, why it matters, and key facts. "
    "Be concise and factual."
)

CURATOR_SYSTEM = (
    "You are a tech news curator. Given article summaries from "
    "Hacker News, produce a brief digest. For each story write "
    "one line with the title and why it matters. Order by importance. "
    "Skip low-quality or redundant stories."
)

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _msg(role: str, text: str) -> dict:
    return {"role": role, "content": [{"type": "text", "text": text}]}


def _generate(model, system: str, user_msg: str, temp: float, max_tokens: int) -> tuple[str, float]:
    sampler = make_sampler(temp=temp, top_p=0.9, top_k=20)
    t0 = time.perf_counter()
    response = model.generate(
        [_msg("system", system), _msg("user", user_msg)],
        max_tokens=max_tokens,
        sampler=sampler,
    )
    elapsed = time.perf_counter() - t0
    return (response.content or "").strip(), elapsed


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def summarize_story(model, story: Story, max_article_chars: int) -> StoryWithSummary:
    content = get_story_content(story, max_article_chars)
    user_msg = (
        f"Article title: \"{story.title}\"\n"
        f"Content:\n{content}\n\n"
        f"Summarize in 2-3 sentences."
    )
    summary, elapsed = _generate(model, SUMMARIZER_SYSTEM, user_msg, temp=0.3, max_tokens=150)
    return StoryWithSummary(
        story=story,
        content_len=len(content),
        summary=summary,
        elapsed_s=elapsed,
    )


def curate(model, summaries: list[StoryWithSummary]) -> tuple[str, float]:
    entries = []
    for s in summaries:
        entries.append(
            f"- \"{s.story.title}\" (▲{s.story.score}, {s.story.comments} comments)\n"
            f"  Summary: {s.summary}"
        )
    user_msg = (
        f"{len(entries)} Hacker News stories:\n\n"
        + "\n\n".join(entries)
        + "\n\nProduce the digest."
    )
    return _generate(model, CURATOR_SYSTEM, user_msg, temp=0.3, max_tokens=500)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_digest(model, stories_count: int = 5, max_article_chars: int = 3000):
    t_start = time.perf_counter()

    banner(f"HN DIGEST — top {stories_count}")

    console.print()
    console.print("  Fetching from Hacker News …", style="dim")
    stories = fetch_top_stories(stories_count)
    console.print(f"  {len(stories)} stories.", style="green")
    console.print()

    table = Table(show_header=False, box=None, padding=(0, 1), pad_edge=False)
    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("Title", no_wrap=False)
    table.add_column("Stats", style="dim", width=16, justify="right")
    for i, s in enumerate(stories, 1):
        domain = f" [dim]({urlparse(s.url).netloc})[/dim]" if s.url else ""
        table.add_row(str(i), f"{s.title}{domain}", f"▲ {s.score}  💬 {s.comments}")
    console.print(table)

    # -- summarize ---------------------------------------------------
    section("Summarizing")

    summaries = []
    for story in stories:
        result = summarize_story(model, story, max_article_chars)
        summaries.append(result)
        chars = f"{result.content_len} chars" if result.content_len > len(story.title) else "title only"
        console.print()
        console.print(Panel(
            result.summary.strip(),
            title=f"[cyan]{result.story.title}[/cyan]",
            subtitle=f"[dim]{chars}, {result.elapsed_s:.1f}s[/dim]",
            expand=True,
            padding=(1, 2),
        ))

    # -- curate ------------------------------------------------------
    banner("DIGEST")

    digest, c_elapsed = curate(model, summaries)
    print_turn("Curator", "yellow", digest, c_elapsed, icon="◆")

    total = time.perf_counter() - t_start
    stats_footer(f"{len(summaries)} stories  ·  total {total:.1f}s")

    return summaries, digest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 02: HN Digest with 1-bit agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        examples:
          python -m experiments.02_hn_digest.run
          python -m experiments.02_hn_digest.run --stories 10
          python -m experiments.02_hn_digest.run --stories 3 --model 1.7b
        """),
    )
    parser.add_argument(
        "--stories", type=int, default=5,
        help="number of stories to process (default: 5)",
    )
    parser.add_argument(
        "--model", default="8b",
        help="model size (default: 8b)",
    )
    parser.add_argument(
        "--max-article-chars", type=int, default=3000,
        help="max chars to extract per article (default: 3000)",
    )
    args = parser.parse_args()

    model = load_model(args.model)
    run_digest(model, args.stories, args.max_article_chars)


if __name__ == "__main__":
    main()
