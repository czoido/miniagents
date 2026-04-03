"""Simulation logging — writes a markdown file per run."""

from datetime import datetime
from pathlib import Path

from .world import Action, Citizen, Event, Premise

_LOG_DIR = Path(__file__).parent / "logs"


class SimLog:
    """Accumulates simulation data and writes a markdown log file."""

    def __init__(self, days: int, hours: int, seed: int | None):
        self._lines: list[str] = []
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self._path = _LOG_DIR / f"{ts}.md"
        self._add(f"# Mini City — {days} day(s), {hours} hours/day, seed={seed}\n")

    def _add(self, text: str) -> None:
        self._lines.append(text)

    def premise(self, p: Premise) -> None:
        self._add(f"\n## Setting\n")
        self._add(f"**{p.village}** — {p.region} ({p.era})")
        self._add(f"*{p.mood}*")
        self._add(f"Store: {p.store}\n")

    def finances(self, citizens: list[Citizen], label: str) -> None:
        self._add(f"\n## {label}\n")
        for c in citizens:
            inc = f"  (+${c.income}/day)" if c.income else ""
            self._add(f"- **{c.name}** ({c.role}): ${c.money}{inc}")
            if c.secret:
                self._add(f"  - *Secret: {c.secret}*")
            if c.goals:
                for g in c.goals:
                    self._add(f"  - {g}")
            if c.relationships:
                for other, rel in c.relationships.items():
                    self._add(f"  - → {other}: {rel}")

    def event(self, text: str) -> None:
        self._add(f"\n> **Breaking News:** {text}\n")

    def day_header(self, day: int, total: int) -> None:
        self._add(f"\n---\n\n# Day {day} / {total}\n")

    def interaction(self, ev: Event) -> None:
        names = " & ".join(ev.participants)
        self._add(f"\n### {ev.time_label} at {ev.location} — {names}\n")
        for name, line in ev.transcript:
            self._add(f"**{name}:** {line}\n")
        self._add(f"*Summary: {ev.summary}*\n")

    def actions(self, action_list: list[Action], citizens: list[Citizen]) -> None:
        cmap = {c.name: c for c in citizens}
        self._add("\n### End of Day — Actions\n")
        for a in action_list:
            c = cmap[a.citizen]
            delta = ""
            if a.cost > 0:
                delta = f" (-${a.cost}, ${c.money} left)"
            elif a.earned > 0:
                delta = f" (+${a.earned}, ${c.money} now)"
            self._add(f"- **{a.citizen}:** {a.description}{delta}")

    def chronicle(self, text: str, day: int | None = None) -> None:
        label = f"Day {day} Chronicle" if day else "Chronicle"
        self._add(f"\n### {label}\n\n{text}\n")

    def overnight(self, citizens: list[Citizen]) -> None:
        self._add("\n### Overnight\n")
        for c in citizens:
            goals = ", ".join(c.goals[:2]) or "no clear goals"
            self._add(f"- **{c.name}** (${c.money}): {goals}")

    def final_recap(self, text: str) -> None:
        self._add(f"\n---\n\n## Final Recap\n\n{text}\n")

    def write(self, total_s: float) -> Path:
        self._add(f"\n---\n*Simulation completed in {total_s:.1f}s*\n")
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        self._path.write_text("\n".join(self._lines), encoding="utf-8")
        return self._path
