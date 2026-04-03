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

from shared.models import MODELS, load_model

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HN_API = "https://hacker-news.firebaseio.com/v0"
W = 64

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
# Fetching (plain Python — no agent needed for HTTP plumbing)
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
# Display
# ---------------------------------------------------------------------------


def _print_wrapped(text: str, indent: int = 4):
    prefix = " " * indent
    for line in text.splitlines():
        print(textwrap.fill(line, width=W, initial_indent=prefix, subsequent_indent=prefix))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_digest(model, stories_count: int = 5, max_article_chars: int = 3000):
    t_start = time.perf_counter()

    print(f"\n{'═' * W}")
    print(f"  HN DIGEST — top {stories_count}")
    print(f"{'═' * W}")

    # -- fetch -------------------------------------------------------
    print(f"\n  Fetching from Hacker News...")
    stories = fetch_top_stories(stories_count)
    print(f"  {len(stories)} stories.\n")

    for i, s in enumerate(stories, 1):
        domain = f" ({urlparse(s.url).netloc})" if s.url else ""
        print(f"  {i:2d}. {s.title}{domain}")
        print(f"      ▲ {s.score}  💬 {s.comments}")

    # -- summarize ---------------------------------------------------
    print(f"\n{'─' * W}")
    print(f"  Summarizing...")
    print(f"{'─' * W}")

    summaries = []
    for story in stories:
        result = summarize_story(model, story, max_article_chars)
        summaries.append(result)
        chars = f"{result.content_len} chars" if result.content_len > len(story.title) else "title only"
        print(f"\n  \033[36m● {result.story.title}\033[0m")
        print(f"    [{chars}, {result.elapsed_s:.1f}s]\n")
        _print_wrapped(result.summary)

    # -- curate ------------------------------------------------------
    print(f"\n{'═' * W}")
    print(f"  DIGEST")
    print(f"{'═' * W}")

    digest, c_elapsed = curate(model, summaries)
    print(f"\n  \033[33m◆ Curator\033[0m  ({c_elapsed:.1f}s)\n")
    _print_wrapped(digest)

    total = time.perf_counter() - t_start
    print(f"\n{'─' * W}")
    print(f"  {len(summaries)} stories  ·  total {total:.1f}s")
    print(f"{'═' * W}\n")

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
        "--model", choices=MODELS, default="8b",
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
