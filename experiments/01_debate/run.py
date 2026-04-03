"""
Experiment 01 — Adversarial Debate

Multiple 1-bit model instances take opposing sides on a proposition and
debate across several rounds. A judge reads the full transcript and
delivers a synthesized verdict.

Hypothesis: structured disagreement between cheap, fast models produces
better-reasoned outputs than a single model thinking alone.

Usage:
  python -m experiments.01_debate.run "Should we colonize Mars?"
  python -m experiments.01_debate.run "Is TDD worth the overhead?" --rounds 4
  python -m experiments.01_debate.run --interactive --judge-model 8b
"""

import argparse
import re
import textwrap
import time
from dataclasses import dataclass

from mlx_lm.sample_utils import make_sampler

from shared.console import banner, console, print_turn, section, stats_footer
from shared.models import MODELS, load_model

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class Turn:
    speaker: str
    round_num: int
    content: str
    elapsed_s: float


# ---------------------------------------------------------------------------
# Debater & judge profiles
# ---------------------------------------------------------------------------

DEBATERS = {
    "advocate": {
        "name": "Advocate",
        "style": "green",
        "system": (
            "You argue IN FAVOR of any proposition given to you. "
            "You are optimistic and focus on benefits and opportunities. "
            "Give 2-3 specific points. Under 100 words. "
            "NEVER argue against the proposition."
        ),
        "temp": 0.8,
    },
    "critic": {
        "name": "Critic",
        "style": "red",
        "system": (
            "You argue AGAINST any proposition given to you. "
            "You are skeptical and focus on risks, costs, and weak evidence. "
            "Give 2-3 specific points. Under 100 words. "
            "NEVER agree with or support the proposition."
        ),
        "temp": 0.5,
    },
}

JUDGE = {
    "name": "Judge",
    "style": "yellow",
    "icon": "◆",
    "system": (
        "You are an impartial judge. Read the debate transcript. "
        "State which side won and why in 2-3 sentences. "
        "Then name the single best argument from each side."
    ),
    "temp": 0.3,
}

# ---------------------------------------------------------------------------
# Output cleaning — small models leak prompt fragments into responses
# ---------------------------------------------------------------------------

_LEAK_PATTERNS = [
    re.compile(r"^\s*\*{0,2}\s*\[?\s*round\s+\d+[^]]*\]?\s*\*{0,2}\s*[-—:]?\s*", re.I),
    re.compile(r"^\s*\*{0,2}\s*counter[- ]?argument\s*\*{0,2}\s*[-—:]?\s*", re.I),
    re.compile(r"^\s*\*{0,2}\s*respond\s+to\s+the\s+latest.*$", re.I | re.M),
    re.compile(r"^\s*[-—]+\s*$", re.M),
]


def _clean(text: str) -> str:
    text = text.strip()
    for pat in _LEAK_PATTERNS:
        text = pat.sub("", text).strip()
    return text


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _msg(role: str, text: str) -> dict:
    return {"role": role, "content": [{"type": "text", "text": text}]}


def _generate(model, system: str, user_msg: str, temp: float, max_tokens: int):
    messages = [_msg("system", system), _msg("user", user_msg)]
    sampler = make_sampler(temp=temp, top_p=0.9, top_k=20)
    t0 = time.perf_counter()
    response = model.generate(messages, max_tokens=max_tokens, sampler=sampler)
    elapsed = time.perf_counter() - t0
    return _clean(response.content or ""), elapsed


# ---------------------------------------------------------------------------
# Debate engine
# ---------------------------------------------------------------------------


def _format_transcript(turns: list[Turn]) -> str:
    return "\n".join(f"[{t.speaker}]\n{t.content.strip()}\n" for t in turns)


def _last_opponent_turn(turns: list[Turn], current_speaker: str) -> Turn | None:
    for t in reversed(turns):
        if t.speaker != current_speaker:
            return t
    return None


def run_debate(
    topic: str,
    model,
    judge_model=None,
    *,
    rounds: int = 3,
    max_tokens: int = 200,
    max_tokens_judge: int = 400,
):
    jmodel = judge_model or model
    turns: list[Turn] = []
    t_start = time.perf_counter()

    banner(f"ADVERSARIAL DEBATE — {rounds} rounds", subtitle=f'"{topic}"')

    order = list(DEBATERS.keys())

    for rnd in range(1, rounds + 1):
        section(f"Round {rnd}/{rounds}")

        for key in order:
            d = DEBATERS[key]

            if not turns:
                user_msg = (
                    f"Proposition: \"{topic}\"\n\n"
                    f"Give your opening argument."
                )
            else:
                opponent = _last_opponent_turn(turns, d["name"])
                if opponent:
                    user_msg = (
                        f"Proposition: \"{topic}\"\n\n"
                        f"Your opponent ({opponent.speaker}) just said:\n"
                        f"\"{opponent.content.strip()}\"\n\n"
                        f"Respond to their specific points."
                    )
                else:
                    user_msg = (
                        f"Proposition: \"{topic}\"\n\n"
                        f"Continue your argument."
                    )

            text, elapsed = _generate(
                model, d["system"], user_msg, d["temp"], max_tokens
            )
            print_turn(d["name"], d["style"], text, elapsed)
            turns.append(Turn(d["name"], rnd, text, elapsed))

    # ------------------------------------------------------------------
    # Judge verdict
    # ------------------------------------------------------------------
    banner("VERDICT")

    transcript = _format_transcript(turns)
    judge_msg = (
        f"Proposition: \"{topic}\"\n\n"
        f"Debate transcript:\n{transcript}\n"
        f"Which side won and why?"
    )
    verdict, v_elapsed = _generate(
        jmodel, JUDGE["system"], judge_msg, JUDGE["temp"], max_tokens_judge
    )
    print_turn(JUDGE["name"], JUDGE["style"], verdict, v_elapsed, icon=JUDGE["icon"])

    total = time.perf_counter() - t_start
    n = len(turns)
    avg = sum(t.elapsed_s for t in turns) / n if n else 0

    stats_footer(f"{n} turns  ·  avg {avg:.1f}s/turn  ·  total {total:.1f}s")

    return turns, verdict


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 01: Adversarial Debate with 1-bit models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        examples:
          python -m experiments.01_debate.run "Should we colonize Mars?"
          python -m experiments.01_debate.run "Is TDD worth it?" --rounds 4
          python -m experiments.01_debate.run --interactive --judge-model 8b
        """),
    )
    parser.add_argument("topic", nargs="?", help="proposition to debate")
    parser.add_argument(
        "--rounds", type=int, default=3, help="number of debate rounds (default: 3)"
    )
    parser.add_argument(
        "--model",
        choices=MODELS,
        default="8b",
        help="model size for debaters (default: 8b)",
    )
    parser.add_argument(
        "--judge-model",
        choices=MODELS,
        default=None,
        help="model size for judge (defaults to --model)",
    )
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--max-tokens-judge", type=int, default=400)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    model = load_model(args.model)
    jmodel = load_model(args.judge_model) if args.judge_model and args.judge_model != args.model else None

    run_kwargs = dict(
        rounds=args.rounds,
        max_tokens=args.max_tokens,
        max_tokens_judge=args.max_tokens_judge,
    )

    if args.interactive:
        console.print("Interactive debate mode. Type a proposition or 'quit' to exit.\n")
        while True:
            try:
                topic = input("Proposition: ").strip()
            except (KeyboardInterrupt, EOFError):
                console.print()
                break
            if not topic or topic.lower() in ("quit", "exit", "q"):
                break
            run_debate(topic, model, jmodel, **run_kwargs)
    elif args.topic:
        run_debate(args.topic, model, jmodel, **run_kwargs)
    else:
        parser.error("provide a topic or use --interactive")


if __name__ == "__main__":
    main()
