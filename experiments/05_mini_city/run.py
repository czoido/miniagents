"""
Experiment 05 — Mini City

A village simulation where citizen-agents with distinct personalities
interact freely across one or more days. Each citizen has money, goals,
and a daily income from their job. A general store sells anything — if
you have the cash, you can buy what you need to complete your goals.
Random events shake things up each morning.

Each hour maps to a real time of day with locations and atmosphere.
Between days, citizens sleep: memories compress and goals evolve.
A factual narrator summarizes each day.

Tests whether 1-bit models can:
  - Maintain consistent personality and manage resources
  - Complete goals through conversation, spending, and work
  - React to injected random events and economic pressure
  - Sustain character continuity across multiple days
  - Produce emergent narrative through unscripted interaction

Usage:
  python -m experiments.05_mini_city.run
  python -m experiments.05_mini_city.run --days 3 --hours 6
  python -m experiments.05_mini_city.run --citizens 8 --days 3
  python -m experiments.05_mini_city.run --days 7 --hours 4 --model 8b
"""

import argparse
import random
import textwrap
import time

from rich.panel import Panel
from rich.text import Text

from shared.console import agent_tag, banner, console, section, stats_footer
from shared.models import MODELS, load_model

from .engine import (
    day_chronicle,
    form_groups,
    generate,
    generate_citizens,
    generate_event,
    generate_premise,
    generate_schedule,
    overnight,
    resolve_daily_actions,
    run_interaction,
)
from .log import SimLog
from .prompts import FINAL_SUMMARY_SYSTEM
from .world import CITIZENS, Citizen, Event


# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------


def run_city(
    model,
    *,
    days: int = 1,
    hours: int = 6,
    max_tokens: int = 150,
    seed: int | None = None,
    num_citizens: int = 5,
    setup: str = "",
):
    rng = random.Random(seed)

    # -- Generate world premise --
    section("Generating World")
    premise = generate_premise(model, user_setup=setup)
    console.print(f"  [bold]{premise.village}[/bold]")
    console.print(f"  [dim]{premise.region} · {premise.era}[/dim]")
    console.print(f"  [italic]{premise.mood}[/italic]")
    console.print(f"  [dim]Store: {premise.store}[/dim]")
    console.print()

    # -- Generate citizens --
    num_citizens = max(2, num_citizens)
    section("Generating Citizens")
    console.print(f"  [dim]Creating {num_citizens} unique villagers...[/dim]")
    console.print()
    citizens = generate_citizens(model, num_citizens, premise, user_setup=setup)
    if len(citizens) < 2:
        console.print("  [red]Not enough citizens generated. Using defaults.[/red]")
        citizens = [Citizen(**c.__dict__) for c in CITIZENS]
    else:
        for c in citizens:
            console.print(
                f"  [{c.style}]{c.name}[/{c.style}] — {c.role} "
                f"(${c.money}, +${c.income}/day)"
            )
            # Show personality without the "You are X..." prefix
            parts = c.personality.split(".", 2)
            desc = parts[2].strip() if len(parts) > 2 else parts[-1].strip()
            console.print(f"    [dim]{desc}[/dim]")
            if c.secret:
                console.print(f"    [red dim]Secret: {c.secret}[/red dim]")
        console.print()

        # Show relationships
        any_rels = any(c.relationships for c in citizens)
        if any_rels:
            section("Relationships")
            for c in citizens:
                for other, rel in c.relationships.items():
                    console.print(
                        f"  [{c.style}]{c.name}[/{c.style}] → {other}: "
                        f"[dim]{rel}[/dim]"
                    )
            console.print()

    citizen_map = {c.name: c for c in citizens}
    starting_money = {c.name: c.money for c in citizens}
    all_events: list[Event] = []
    daily_chronicles: list[str] = []
    used_events: list[str] = []
    log = SimLog(days, hours, seed)
    log.premise(premise)
    t_start = time.perf_counter()

    # -- Generate schedule --
    section("Generating Locations")
    full_schedule = generate_schedule(model, premise)
    schedule = full_schedule[:hours]
    for slot in schedule:
        console.print(f"  [dim]{slot.time}[/dim]  {slot.location}")
    console.print()

    names = ", ".join(f"[{c.style}]{c.name}[/{c.style}]" for c in citizens)

    day_label = f"{days} day{'s' if days > 1 else ''}"
    banner(
        f"{premise.village.upper()} — {day_label}, "
        f"{schedule[0].time}–{schedule[-1].time}, "
        f"{len(citizens)} citizens",
        subtitle=names,
    )

    section("Starting Finances")
    for c in citizens:
        inc = f"  +${c.income}/day" if c.income else ""
        console.print(
            f"  [{c.style}]{c.name}[/{c.style}] ({c.role}): "
            f"[bold]${c.money}[/bold]{inc}"
        )
    console.print()
    log.finances(citizens, "Starting Finances")

    for day in range(1, days + 1):
        day_events: list[Event] = []

        if days > 1:
            banner(f"Day {day} / {days}")
            log.day_header(day, days)

        # -- Random event --
        day_event_text = generate_event(model, citizens, used_events, rng, premise)
        used_events.append(day_event_text)

        section("Breaking News")
        console.print(f"  [bold yellow]![/bold yellow] {day_event_text}")
        console.print()
        log.event(day_event_text)
        for c in citizens:
            c.memory.append(f"(news) {day_event_text}")

        # -- Hourly interactions --
        discussed_topics: list[str] = []
        for h, slot in enumerate(schedule, 1):
            day_prefix = f"[dim]Day {day}[/dim]  " if days > 1 else ""
            section(f"{day_prefix}{slot.time}  ·  {slot.period}")
            console.print(f"  [dim italic]{slot.atmosphere}[/dim italic]")
            console.print()

            groups = form_groups(model, citizens, rng)
            for group in groups:
                names_str = ", ".join(c.name for c in group)
                console.print(f"  [dim]→ {names_str} meet up[/dim]")

            console.print()

            for group in groups:
                event = run_interaction(
                    model, group, h, slot, max_tokens,
                    day_event=day_event_text,
                    discussed_topics=discussed_topics,
                    premise=premise,
                )
                day_events.append(event)
                all_events.append(event)
                discussed_topics.append(event.summary)
                _display_interaction(event, citizen_map)
                log.interaction(event)

            _spread_gossip(citizens, groups, day_events, h, rng)

        # -- Daily actions --
        section("End of Day — Actions")
        actions = resolve_daily_actions(model, citizens, rng)
        log.actions(actions, citizens)
        for act in actions:
            c = citizen_map[act.citizen]
            income_tag = f"[green]+${act.earned}[/green] " if act.earned > 0 else ""
            if act.cost > 0:
                console.print(
                    f"  [{c.style}]{c.name}[/{c.style}] "
                    f"{income_tag}[red]-${act.cost}[/red]  {act.description}  "
                    f"[dim](${c.money} left)[/dim]"
                )
            elif act.earned > 0:
                console.print(
                    f"  [{c.style}]{c.name}[/{c.style}] "
                    f"{income_tag} {act.description}  "
                    f"[dim](${c.money} now)[/dim]"
                )
            else:
                console.print(
                    f"  [{c.style}]{c.name}[/{c.style}] "
                    f"[dim]{act.description}[/dim]"
                )
        console.print()

        # -- Day chronicle --
        chronicle, narr_elapsed = day_chronicle(
            model, citizens, day_events, day, actions=actions,
        )
        daily_chronicles.append(chronicle)

        day_title = f"Day {day} Summary" if days > 1 else "Daily Summary"
        console.print(Panel(
            chronicle,
            title=f"[bold]{day_title}[/bold]",
            subtitle=f"[dim]Narrator  ({narr_elapsed:.1f}s)[/dim]",
            expand=True,
            padding=(1, 2),
        ))
        log.chronicle(chronicle, day=day)

        # -- Overnight (skip after last day) --
        if day < days:
            section("Overnight")
            console.print("  [dim italic]The village sleeps...[/dim italic]")
            overnight(model, citizens, day_events)

            for c in citizens:
                goals_str = ", ".join(c.goals[:2]) or "no clear goals"
                console.print(
                    f"  [{c.style}]{c.name}[/{c.style}] "
                    f"[bold]${c.money}[/bold]  "
                    f"[dim]→ {goals_str}[/dim]"
                )
            console.print()
            log.overnight(citizens)

    # -- Final summary (multi-day) --
    if days > 1:
        banner("Final Summary")
        all_chronicles = "\n\n".join(
            f"Day {i+1}:\n{ch}" for i, ch in enumerate(daily_chronicles)
        )
        prompt = f"Daily reports:\n{all_chronicles}\n\nWrite the final recap."
        final, final_elapsed = generate(
            model, FINAL_SUMMARY_SYSTEM, prompt, 0.3, 600,
        )
        console.print(Panel(
            final.strip(),
            title=f"[bold]{days}-Day Recap[/bold]",
            subtitle=f"[dim]({final_elapsed:.1f}s)[/dim]",
            expand=True,
            padding=(1, 2),
        ))
        log.final_recap(final.strip())

    # -- Final finances --
    section("Final Finances")
    for c in citizens:
        delta = c.money - starting_money[c.name]
        sign = f"+${delta}" if delta >= 0 else f"-${abs(delta)}"
        color = "green" if delta >= 0 else "red"
        console.print(
            f"  [{c.style}]{c.name}[/{c.style}]: "
            f"[bold]${c.money}[/bold]  [{color}]({sign})[/{color}]"
        )
    console.print()

    log.finances(citizens, "Final Finances")

    total = time.perf_counter() - t_start
    log_path = log.write(total)
    stats_footer(
        f"{len(all_events)} interactions  ·  {days} day{'s' if days > 1 else ''}  ·  "
        f"{schedule[0].time}–{schedule[-1].time}  ·  "
        f"{len(citizens)} citizens  ·  total {total:.1f}s"
    )
    console.print(f"  [dim]Log saved → {log_path}[/dim]")
    console.print()


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _display_interaction(event: Event, citizen_map: dict[str, Citizen]) -> None:
    turns = len(event.transcript)
    parts = []
    for i, name in enumerate(event.participants):
        c = citizen_map.get(name)
        if not c:
            continue
        if i > 0:
            parts.append(("  ×  ", ""))
        parts.append(("  ", ""))
        parts.append(agent_tag(name, c.style))
        parts.append((f" ${c.money}", ""))
    parts.append((f"  ({turns} turns, {event.elapsed_s:.1f}s)  ", ""))
    title = Text.assemble(*parts)

    dialogue = Text()
    for i, (name, line) in enumerate(event.transcript):
        c = citizen_map.get(name)
        style = f"bold {c.style}" if c else "bold"
        if i > 0:
            dialogue.append("\n\n")
        dialogue.append(f"{name}: ", style=style)
        dialogue.append(line)

    console.print(Panel(
        dialogue,
        title=title,
        subtitle=f"[dim]{event.location} — {event.summary}[/dim]",
        expand=True,
        padding=(1, 2),
    ))


def _spread_gossip(
    citizens: list[Citizen],
    groups: list[list[Citizen]],
    day_events: list[Event],
    hour: int,
    rng: random.Random,
) -> None:
    involved: set[str] = set()
    for group in groups:
        involved.update(c.name for c in group)
    hour_summaries = [e.summary for e in day_events if e.hour == hour]
    for c in citizens:
        if c.name not in involved and hour_summaries:
            gossip = rng.choice(hour_summaries)
            c.memory.append(f"(heard) {gossip}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 05: Mini City — village simulation with 1-bit agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        examples:
          python -m experiments.05_mini_city.run
          python -m experiments.05_mini_city.run --setup "3 survivors on a desert island after a shipwreck"
          python -m experiments.05_mini_city.run --setup "medieval village, 6 characters, a murder mystery"
          python -m experiments.05_mini_city.run --citizens 8 --days 3
          python -m experiments.05_mini_city.run --seed 42
        """),
    )
    parser.add_argument(
        "--setup", type=str, default=None,
        help="describe the scenario (characters, setting, situation) — guided generation",
    )
    parser.add_argument(
        "--citizens", type=int, default=None,
        help="number of citizens to generate (default: 5, min: 4)",
    )
    parser.add_argument(
        "--days", type=int, default=1,
        help="days to simulate (default: 1)",
    )
    parser.add_argument(
        "--hours", type=int, default=6,
        help="time slots per day (default: 6, max 10)",
    )
    parser.add_argument(
        "--model", default="qwen-7b",
        help=f"model key ({', '.join(MODELS)}) or HuggingFace ID (default: qwen-7b)",
    )
    parser.add_argument("--max-tokens", type=int, default=150)
    parser.add_argument(
        "--seed", type=int, default=None,
        help="random seed for reproducibility",
    )
    args = parser.parse_args()

    setup = args.setup
    if setup is None:
        from rich.prompt import Prompt
        console.print()
        setup = Prompt.ask(
            "  [bold]Describe your scenario[/bold] "
            "[dim](or press Enter for a random world)[/dim]\n ",
            default="",
            console=console,
        )

    num_citizens = args.citizens
    if num_citizens is None and setup:
        import re
        words = (
            r"characters|people|survivors|persons|citizens|villagers|"
            r"men|women|boys|girls|friends|strangers|agents|"
            r"personajes|personas|supervivientes|hombres|mujeres|amigos|habitantes"
        )
        nums = re.findall(rf"(\d+)\s*(?:{words})", setup, re.IGNORECASE)
        if not nums:
            nums = re.findall(r"(\d+)\s+\w+", setup)
        num_citizens = int(nums[0]) if nums else 5
    num_citizens = num_citizens or 5

    model = load_model(args.model)
    run_city(
        model,
        days=args.days,
        hours=args.hours,
        max_tokens=args.max_tokens,
        seed=args.seed,
        num_citizens=num_citizens,
        setup=setup,
    )


if __name__ == "__main__":
    main()
