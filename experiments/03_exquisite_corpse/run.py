"""
Experiment 03 — Exquisite Corpse

Collaborative storytelling inspired by the surrealist game. Multiple
agents with distinct writing styles take turns continuing a story, but
each only sees the last few sentences from the previous turn.

Rules:
  1. An agent writes a passage continuing the story
  2. Only the last N sentences are revealed to the next agent
  3. The next agent (different voice) continues from those sentences
  4. Repeat for K turns
  5. The full story is revealed at the end

The less overlap, the more surreal the result.

Usage:
  python -m experiments.03_exquisite_corpse.run
  python -m experiments.03_exquisite_corpse.run --seed "The door opened slowly"
  python -m experiments.03_exquisite_corpse.run --turns 10 --overlap 1
  python -m experiments.03_exquisite_corpse.run --overlap 1 --model 8b
"""

import argparse
import re
import textwrap
import time
from dataclasses import dataclass

from mlx_lm.sample_utils import make_sampler

from shared.models import MODELS, load_model

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class Fragment:
    voice: str
    saw: str
    wrote: str
    elapsed_s: float


# ---------------------------------------------------------------------------
# Voices — each agent has a distinct writing personality
# ---------------------------------------------------------------------------

_STORY_RULES = (
    "You are a collaborative storyteller. You receive a fragment of a story "
    "and must continue it with exactly 2-3 sentences.\n"
    "RULES:\n"
    "- Advance the PLOT: introduce an event, action, or decision.\n"
    "- Keep or develop any characters mentioned. Add a new character only if it serves the plot.\n"
    "- Do NOT just describe atmosphere. Something must HAPPEN.\n"
    "- Do NOT repeat or rephrase what you were given.\n"
    "- Write ONLY your continuation, nothing else."
)

VOICES = [
    {
        "name": "Poet",
        "tag": "\033[35m● Poet\033[0m",
        "system": (
            f"{_STORY_RULES}\n"
            "YOUR VOICE: lyrical and metaphorical, but always in service of the narrative."
        ),
        "temp": 0.9,
    },
    {
        "name": "Noir",
        "tag": "\033[90m● Noir\033[0m",
        "system": (
            f"{_STORY_RULES}\n"
            "YOUR VOICE: hard-boiled noir. Short punchy sentences. Cynical tone."
        ),
        "temp": 0.7,
    },
    {
        "name": "Sci-Fi",
        "tag": "\033[36m● Sci-Fi\033[0m",
        "system": (
            f"{_STORY_RULES}\n"
            "YOUR VOICE: speculative sci-fi. Technical details grounded in the scene."
        ),
        "temp": 0.8,
    },
    {
        "name": "Absurdist",
        "tag": "\033[33m● Absurdist\033[0m",
        "system": (
            f"{_STORY_RULES}\n"
            "YOUR VOICE: surreal and unexpected, but your twist must connect to the story."
        ),
        "temp": 0.95,
    },
]

DEFAULT_SEEDS = [
    "The clock struck thirteen and the shadows began to move on their own.",
    "She found the letter inside a book she had never opened.",
    "The last train of the night carried only one passenger and no driver.",
    "When the music stopped, everyone in the room had aged ten years.",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

W = 64


def _msg(role: str, text: str) -> dict:
    return {"role": role, "content": [{"type": "text", "text": text}]}


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences (handles ., !, ?, and ellipsis)."""
    parts = re.split(r'(?<=[.!?…])\s+', text.strip())
    return [s for s in parts if s]


def _tail_sentences(text: str, n: int) -> str:
    """Return the last *n* sentences of *text*."""
    sentences = _split_sentences(text)
    return " ".join(sentences[-n:]) if len(sentences) > n else text


def _generate(model, system: str, user_msg: str, temp: float, max_tokens: int) -> tuple[str, float]:
    sampler = make_sampler(temp=temp, top_p=0.95, top_k=30)
    t0 = time.perf_counter()
    response = model.generate(
        [_msg("system", system), _msg("user", user_msg)],
        max_tokens=max_tokens,
        sampler=sampler,
    )
    elapsed = time.perf_counter() - t0
    return (response.content or "").strip(), elapsed


def _print_wrapped(text: str, indent: int = 4):
    prefix = " " * indent
    for line in text.splitlines():
        print(textwrap.fill(line, width=W, initial_indent=prefix, subsequent_indent=prefix))


# ---------------------------------------------------------------------------
# Game
# ---------------------------------------------------------------------------


def run_exquisite_corpse(
    model,
    *,
    seed: str | None = None,
    turns: int = 6,
    overlap: int = 2,
    max_tokens: int = 150,
):
    import random

    if seed is None:
        seed = random.choice(DEFAULT_SEEDS)

    fragments: list[Fragment] = []
    t_start = time.perf_counter()

    print(f"\n{'═' * W}")
    print(f"  EXQUISITE CORPSE — {turns} turns, {overlap}-sentence overlap")
    print(f"{'═' * W}")
    print(f"\n  \033[1mSeed:\033[0m \"{seed}\"\n")

    previous_text = seed

    for i in range(turns):
        voice = VOICES[i % len(VOICES)]
        visible = _tail_sentences(previous_text, overlap)

        user_msg = (
            f"Here is the latest part of a story:\n\n"
            f"\"{visible}\"\n\n"
            f"Continue the story. What happens next? Write 2-3 sentences."
        )

        text, elapsed = _generate(
            model, voice["system"], user_msg, voice["temp"], max_tokens
        )

        if not text:
            text, elapsed = _generate(
                model, voice["system"], user_msg, voice["temp"], max_tokens
            )

        fragments.append(Fragment(
            voice=voice["name"],
            saw=visible,
            wrote=text,
            elapsed_s=elapsed,
        ))

        print(f"{'─' * W}")
        print(f"  Turn {i + 1}/{turns}  {voice['tag']}  ({elapsed:.1f}s)")
        print(f"  saw: \"{visible}\"")
        print()
        _print_wrapped(text)
        print()

        if text:
            previous_text = text

    # -- assemble full story ------------------------------------------
    parts = [seed]
    for f in fragments:
        if f.wrote:
            parts.append(f.wrote)
    full = "\n\n".join(parts)

    print(f"{'═' * W}")
    print(f"  FULL STORY")
    print(f"{'═' * W}\n")
    _print_wrapped(full)

    # -- critic: evaluate --------------------------------------------
    print(f"\n{'═' * W}")
    print(f"  \033[32m● Critic\033[0m — evaluating …")
    print(f"{'═' * W}\n")

    eval_system = (
        "You are a literary critic. You receive a short story written "
        "collaboratively by multiple authors who could only see a fragment "
        "of what came before.\n"
        "Rate three aspects on a 1-10 scale:\n"
        "- Narrative coherence\n"
        "- Creativity\n"
        "- Style transitions\n"
        "Then note the strongest moment and the weakest seam.\n"
        "Be concise (5-8 lines max)."
    )

    evaluation, eval_elapsed = _generate(
        model, eval_system,
        f"Here is the story:\n\n{full}",
        temp=0.4, max_tokens=300,
    )

    _print_wrapped(evaluation)
    print(f"\n  \033[32m● Critic\033[0m  ({eval_elapsed:.1f}s)")

    # -- translator --------------------------------------------------
    print(f"\n{'═' * W}")
    print(f"  \033[34m● Translator\033[0m — translating to Spanish …")
    print(f"{'═' * W}\n")

    trans_system = (
        "You are a literary translator. Translate the following English "
        "story into Spanish. Preserve the literary style, rhythm, and tone "
        "of each section. Output ONLY the Spanish translation, nothing else."
    )

    translation, trans_elapsed = _generate(
        model, trans_system,
        full,
        temp=0.3, max_tokens=1200,
    )

    _print_wrapped(translation)
    print(f"\n  \033[34m● Translator\033[0m  ({trans_elapsed:.1f}s)")

    total = time.perf_counter() - t_start
    print(f"\n{'─' * W}")
    voices_used = " → ".join(f.voice for f in fragments)
    print(f"  {voices_used} → Critic → Translator")
    print(f"  {turns} turns  ·  {overlap}-sentence overlap  ·  total {total:.1f}s")
    print(f"{'═' * W}\n")

    return fragments, full, evaluation, translation


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 03: Exquisite Corpse with 1-bit agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        examples:
          python -m experiments.03_exquisite_corpse.run
          python -m experiments.03_exquisite_corpse.run --seed "The door opened slowly"
          python -m experiments.03_exquisite_corpse.run --turns 10 --overlap 1
        """),
    )
    parser.add_argument("--seed", help="opening sentence (random if omitted)")
    parser.add_argument(
        "--turns", type=int, default=6, help="number of turns (default: 6)"
    )
    parser.add_argument(
        "--overlap", type=int, default=2,
        help="sentences visible to next agent (default: 2)",
    )
    parser.add_argument(
        "--model", choices=MODELS, default="8b",
        help="model size (default: 8b)",
    )
    parser.add_argument("--max-tokens", type=int, default=150)
    args = parser.parse_args()

    model = load_model(args.model)
    run_exquisite_corpse(
        model,
        seed=args.seed,
        turns=args.turns,
        overlap=args.overlap,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
