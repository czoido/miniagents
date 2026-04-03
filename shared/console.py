"""Shared Rich console utilities for all experiments."""

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.text import Text

console = Console(width=72)

STYLES = {
    "advocate": "green",
    "critic": "red",
    "judge": "yellow",
    "poet": "magenta",
    "noir": "bright_black",
    "sci-fi": "cyan",
    "absurdist": "yellow",
    "curator": "yellow",
    "translator": "blue",
    "web": "magenta",
    "agent": "cyan",
}


def banner(title: str, subtitle: str | None = None):
    """Major section header — replaces ═ banners."""
    content = title
    if subtitle:
        content += f"\n[dim]{subtitle}[/dim]"
    console.print()
    console.print(Panel(content, style="bold", expand=True))


def section(title: str = ""):
    """Lightweight separator — replaces ─ lines."""
    console.print(Rule(title, style="dim"))


def result_panel(text: str, title: str = "RESULT"):
    """Panel for final output."""
    console.print()
    console.print(Panel(text.strip(), title=f"[bold]{title}[/bold]", expand=True, padding=(1, 2)))
    console.print()


def print_wrapped(text: str, indent: int = 4, style: str = ""):
    """Print text with padding — replaces textwrap.fill loops."""
    pad = " " * indent
    for line in text.strip().splitlines():
        console.print(f"{pad}{line}", style=style, highlight=False)


def agent_tag(name: str, style: str = "cyan", icon: str = "●") -> Text:
    """Colored agent/speaker label."""
    tag = Text()
    tag.append(f"{icon} {name}", style=style)
    return tag


def print_turn(name: str, style: str, text: str, elapsed: float, icon: str = "●"):
    """Print a model turn: tag with timing, then indented content."""
    tag = agent_tag(name, style, icon)
    tag.append(f"  ({elapsed:.1f}s)", style="dim")
    console.print()
    console.print("  ", tag)
    console.print()
    print_wrapped(text)


def print_agent_start(name: str, task: str, style: str = "cyan"):
    """Log a sub-agent starting."""
    preview = task[:120].replace("\n", " ")
    suffix = "…" if len(task) > 120 else ""
    line = Text()
    line.append(f"▶ {name}", style=style)
    line.append(f"  {preview}{suffix}", style="dim")
    console.print()
    console.print("    ", line)


def print_agent_done(name: str, preview: str, style: str = "cyan", extra: str = ""):
    """Log a sub-agent finishing."""
    short = preview[:100].replace("\n", " ").strip()
    suffix = "…" if len(preview) > 100 else ""
    line = Text()
    line.append(f"✔ {name}", style=style)
    if extra:
        line.append(f"  {extra} — ", style="dim")
    else:
        line.append("  ", style="dim")
    line.append(f"{short}{suffix}", style="dim")
    console.print("    ", line)
    console.print()


def print_search_step(name: str, message: str, error: bool = False):
    """Log a web search/fetch step."""
    line = Text()
    line.append(f"🔍 {name}", style="magenta")
    if error:
        line.append(f" {message}", style="red")
    else:
        line.append(f" {message}", style="dim")
    console.print("    ", line)


def code_panel(code: str):
    """Display code in a syntax-highlighted panel."""
    syntax = Syntax(code, "python", theme="monokai", line_numbers=False, padding=1)
    console.print(Panel(syntax, title="[yellow]Code to execute[/yellow]", expand=True))


def stats_footer(text: str):
    """Footer with stats — replaces ─ + stats + ═."""
    section()
    console.print(f"  {text}")
    section()
    console.print()
