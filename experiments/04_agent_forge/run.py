"""
Experiment 04 — Agent Forge

A coordinator agent (smolagents CodeAgent) dynamically discovers sub-agents
defined as markdown files in the agents/ directory. Each .md specifies a
sub-agent's name, description, personality, and config via YAML frontmatter.

The coordinator uses smolagents' full agentic loop: it writes Python code
to call managed agents. Every code snippet requires human approval BEFORE
execution (wrapping the python_executor).

Sub-agents use DirectTextAgent — a lightweight wrapper that calls the model
directly, bypassing the ReAct loop entirely. 1-bit models struggle to produce
the strict <code>final_answer()</code> format that CodeAgent requires, so
direct generation is far more reliable and produces cleaner output.

Tests whether 1-bit models can:
  - Drive the smolagents CodeAgent ReAct loop (coordinator writes Python)
  - Route tasks intelligently to markdown-defined sub-agents
  - Operate within a human-in-the-loop approval workflow

Agent markdown format:

    ---
    name: poet
    description: Writes creative literary prose
    model: 8b          # 8b | 4b | 1.7b (default: 8b)
    temp: 0.9
    max_tokens: 300
    ---

    You are a poet. Write vivid, metaphorical prose...

Usage:
  python -m experiments.04_agent_forge.run "Write a poem about the sea"
  python -m experiments.04_agent_forge.run --auto-approve "Analyze pros and cons of AI"
  python -m experiments.04_agent_forge.run --list
"""

import argparse
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import yaml
from rich.prompt import Prompt
from rich.table import Table
from smolagents import CodeAgent, MLXModel

from shared.console import (
    banner,
    code_panel,
    console,
    print_agent_done,
    print_agent_start,
    print_search_step,
    result_panel,
    section,
)
from shared.models import MODELS

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

AGENT_TYPES = {"text", "web_researcher"}


@dataclass
class AgentDef:
    name: str
    description: str
    instructions: str
    agent_type: str = "text"
    model: str = "8b"
    temp: float = 0.7
    max_tokens: int = 300
    max_sources: int = 3
    source_file: str = ""


# ---------------------------------------------------------------------------
# Markdown loader
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)", re.DOTALL)


def load_agent_defs(agents_dir: str) -> list[AgentDef]:
    """Parse every .md file in *agents_dir* into an AgentDef."""
    dirpath = Path(agents_dir)
    if not dirpath.is_dir():
        raise FileNotFoundError(f"Agents directory not found: {agents_dir}")

    defs: list[AgentDef] = []
    for md in sorted(dirpath.glob("*.md")):
        m = _FRONTMATTER_RE.match(md.read_text())
        if not m:
            continue
        front = yaml.safe_load(m.group(1)) or {}
        body = m.group(2).strip()
        model_size = str(front.get("model", "8b"))
        if model_size not in MODELS:
            raise ValueError(
                f"{md.name}: unknown model '{model_size}' "
                f"(valid: {', '.join(sorted(MODELS))})"
            )
        agent_type = str(front.get("type", "text"))
        if agent_type not in AGENT_TYPES:
            raise ValueError(
                f"{md.name}: unknown type '{agent_type}' "
                f"(valid: {', '.join(sorted(AGENT_TYPES))})"
            )
        defs.append(AgentDef(
            name=front.get("name", md.stem),
            description=front.get("description", ""),
            instructions=body,
            agent_type=agent_type,
            model=model_size,
            temp=float(front.get("temp", 0.7)),
            max_tokens=int(front.get("max_tokens", 300)),
            max_sources=int(front.get("max_sources", 3)),
            source_file=md.name,
        ))
    return defs


# ---------------------------------------------------------------------------
# Pre-execution approval wrapper
# ---------------------------------------------------------------------------


class ApprovingExecutor:
    """Wraps a smolagents python_executor to ask for human approval
    BEFORE running any code."""

    def __init__(self, executor, auto_approve: bool = False):
        self._executor = executor
        self.auto_approve = auto_approve

    def __call__(self, code_action, *args, **kwargs):
        if not self.auto_approve:
            section("Approval required")
            console.print()
            code_panel(code_action)
            console.print()

            while True:
                answer = Prompt.ask(
                    "  Approve?",
                    choices=["y", "n", "q"],
                    default="y",
                )
                if answer == "y":
                    break
                if answer == "n":
                    raise RuntimeError("Denied by user. Try a different approach.")
                if answer == "q":
                    console.print("\n  Aborted.", style="dim")
                    sys.exit(0)

        return self._executor(code_action, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._executor, name)


# ---------------------------------------------------------------------------
# DirectTextAgent — lightweight managed agent (no ReAct loop)
# ---------------------------------------------------------------------------


class DirectTextAgent:
    """Managed agent that calls the model directly instead of using a ReAct loop.

    smolagents' CodeAgent sub-agents require the model to generate strict
    <code>final_answer("...")</code> blocks — 1-bit models can't do this
    reliably.  This class is duck-type compatible with smolagents' managed
    agent interface (name, description, __call__) but skips the code
    generation/parsing entirely.
    """

    def __init__(self, *, model: MLXModel, agent_def: AgentDef):
        self.model = model
        self.name = agent_def.name
        self.description = agent_def.description
        self._instructions = agent_def.instructions
        self._max_tokens = agent_def.max_tokens

    def __call__(self, task: str, **kwargs) -> str:
        print_agent_start(self.name, task, style="cyan")
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self._instructions}]},
            {"role": "user", "content": [{"type": "text", "text": task}]},
        ]
        response = self.model(messages, max_tokens=self._max_tokens)
        text = response.content
        print_agent_done(self.name, text, style="cyan")
        return (
            f"Here is the final answer from your managed agent '{self.name}':\n"
            f"{text}"
        )


# ---------------------------------------------------------------------------
# WebResearchAgent — search + fetch + synthesize pipeline
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r"\[.*?\]\((https?://[^\s)]+)\)")


class WebResearchAgent:
    """Managed agent that searches the web, fetches pages, and synthesizes results.

    Pipeline (no ReAct loop):
      1. DuckDuckGoSearchTool(query) -> markdown with links + snippets
      2. Extract top N URLs from the results
      3. VisitWebpageTool(url) for each -> page content as markdown
      4. Build a prompt with search results + page excerpts + task
      5. Call the model to synthesize an answer
    """

    def __init__(self, *, model: MLXModel, agent_def: AgentDef):
        self.model = model
        self.name = agent_def.name
        self.description = agent_def.description
        self._instructions = agent_def.instructions
        self._max_tokens = agent_def.max_tokens
        self._max_sources = agent_def.max_sources
        self._search = None
        self._visit = None

    def _ensure_tools(self):
        if self._search is not None:
            return
        try:
            from smolagents import DuckDuckGoSearchTool, VisitWebpageTool
        except ImportError as e:
            raise ImportError(
                "Web research requires extra deps: pip install ddgs markdownify requests"
            ) from e
        self._search = DuckDuckGoSearchTool()
        self._visit = VisitWebpageTool()

    def _extract_urls(self, search_result: str) -> list[str]:
        return _URL_RE.findall(search_result)

    def _search_with_retry(self, query: str, max_retries: int = 3) -> str:
        import time
        result = ""
        for attempt in range(max_retries):
            try:
                result = self._search(query)
                if result and len(result.strip()) > 20 and self._extract_urls(result):
                    return result
                if attempt < max_retries - 1:
                    print_search_step(
                        self.name,
                        f"retry {attempt + 1}/{max_retries} (empty results)",
                    )
                    time.sleep(1)
            except Exception as e:
                if attempt == max_retries - 1:
                    return f"Search failed after {max_retries} attempts: {e}"
                time.sleep(1)
        return result or "No search results found."

    def __call__(self, task: str, **kwargs) -> str:
        self._ensure_tools()
        print_agent_start(self.name, task, style="magenta")
        print_search_step(self.name, f"searching: {task[:80]}…")

        search_result = self._search_with_retry(task)

        urls = self._extract_urls(search_result)
        page_excerpts: list[str] = []
        max_chars_per_page = 2000

        for url in urls[: self._max_sources]:
            print_search_step(self.name, f"fetching: {url[:60]}…")
            try:
                content = self._visit(url)
                page_excerpts.append(
                    f"### Source: {url}\n{content[:max_chars_per_page]}"
                )
            except Exception as e:
                print_search_step(self.name, f"fetch failed: {e}", error=True)
                continue

        context_parts = [f"## Search results\n{search_result}"]
        if page_excerpts:
            context_parts.append("## Page contents\n" + "\n\n---\n\n".join(page_excerpts))

        context = "\n\n".join(context_parts)

        system_prompt = (
            f"{self._instructions}\n\n"
            f"You have been given web search results and page contents below. "
            f"Use them to answer the user's question. Cite sources with URLs.\n\n"
            f"{context}"
        )

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": task}]},
        ]
        response = self.model(messages, max_tokens=self._max_tokens)
        text = response.content
        print_agent_done(
            self.name, text, style="magenta",
            extra=f"{len(urls)} sources, {len(page_excerpts)} fetched",
        )
        return (
            f"Here is the final answer from your managed agent '{self.name}':\n"
            f"{text}"
        )


# ---------------------------------------------------------------------------
# Build agents
# ---------------------------------------------------------------------------


def _load_models(defs: list[AgentDef], coordinator_model_size: str) -> dict[str, MLXModel]:
    """Load each unique model size needed (coordinator + sub-agents), reusing instances."""
    sizes = {coordinator_model_size} | {d.model for d in defs}
    models: dict[str, MLXModel] = {}
    for size in sorted(sizes):
        model_id = MODELS[size]
        console.print(f"  Loading [bold]{model_id}[/bold] …")
        models[size] = MLXModel(model_id)
    console.print("  Ready.", style="green")
    console.print()
    return models


def build_managed_agents(
    defs: list[AgentDef], models: dict[str, MLXModel],
) -> list[DirectTextAgent | WebResearchAgent]:
    agents: list[DirectTextAgent | WebResearchAgent] = []
    for d in defs:
        if d.agent_type == "web_researcher":
            agents.append(WebResearchAgent(model=models[d.model], agent_def=d))
        else:
            agents.append(DirectTextAgent(model=models[d.model], agent_def=d))
    return agents


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_forge(
    task: str,
    agents_dir: str,
    model_size: str = "8b",
    auto_approve: bool = False,
):
    defs = load_agent_defs(agents_dir)
    if not defs:
        console.print("  No agent definitions found in", agents_dir, style="red")
        return

    banner(f"AGENT FORGE — {len(defs)} agents discovered")

    table = Table(show_header=False, box=None, padding=(0, 1), pad_edge=False)
    table.add_column("Icon", width=2)
    table.add_column("Name", style="cyan", width=18)
    table.add_column("Description")
    table.add_column("Model", style="dim", width=6, justify="right")
    for d in defs:
        model_tag = d.model if d.model != model_size else ""
        table.add_row("●", d.name, d.description, model_tag)
    console.print(table)
    console.print()

    console.print(f'  [bold]Task:[/bold] "{task}"')

    if not auto_approve:
        console.print()
        console.print("  [yellow]⚠  Human approval is ON[/yellow] — you will review each action before execution.")
    console.print()

    models = _load_models(defs, model_size)
    managed = build_managed_agents(defs, models)

    agent_list = ", ".join(f"`{d.name}`" for d in defs)
    coordinator_instructions = (
        "You delegate tasks to pre-built agents and return their answers.\n\n"
        f"AGENTS YOU CAN CALL: {agent_list}\n\n"
        "RULES:\n"
        "- Agents are already defined. Just CALL them. Your code should be 2-3 lines.\n"
        "- NEVER use `def`. NEVER define functions or classes.\n"
        "- NEVER use `import`. You have no libraries.\n"
        "- NEVER use `print()`. It does nothing useful.\n"
        "- NEVER do math or transform the result. Just pass it through.\n"
        "- ALWAYS end with final_answer(result). Every <code> block MUST have it.\n"
        "- Call an agent: result = agent_name(task=\"description\")\n\n"
        "CORRECT:\n"
        "<code>\n"
        "result = explainer(task=\"What is binary search?\")\n"
        "final_answer(result)\n"
        "</code>\n\n"
        "CORRECT (chaining):\n"
        "<code>\n"
        "info = analyst(task=\"Analyze pros and cons of X\")\n"
        "result = summarizer(task=info)\n"
        "final_answer(result)\n"
        "</code>\n\n"
        "WRONG — NEVER DO THIS:\n"
        "- def explainer(task): ...   # WRONG! Never use def!\n"
        "- import json                # WRONG! Never import!\n"
        "- print(result)              # WRONG! Never use print!\n"
        "- result = 34.75 ** 0.36     # WRONG! Never do math on results!\n"
        "- Final Answer: text         # WRONG! Use final_answer() in code!\n"
    )

    coordinator = CodeAgent(
        tools=[],
        model=models[model_size],
        managed_agents=managed,
        instructions=coordinator_instructions,
        max_steps=12,
        verbosity_level=2,
    )

    coordinator.python_executor = ApprovingExecutor(
        coordinator.python_executor, auto_approve=auto_approve,
    )

    result = coordinator.run(task)

    if result:
        result_panel(str(result))
    else:
        console.print()
        section()
        console.print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    default_agents_dir = str(repo_root / "agents")

    parser = argparse.ArgumentParser(
        description="Experiment 04: Agent Forge — markdown-driven sub-agents via smolagents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        examples:
          python -m experiments.04_agent_forge.run "Write a poem about the sea"
          python -m experiments.04_agent_forge.run --auto-approve "Pros and cons of AI"
          python -m experiments.04_agent_forge.run --list
        """),
    )
    parser.add_argument("task", nargs="?", help="task for the agents")
    parser.add_argument(
        "--agents-dir", default=default_agents_dir,
        help="directory with agent .md definitions (default: agents/)",
    )
    parser.add_argument(
        "--model", default="8b",
        help="model size for coordinator (default: 8b)",
    )
    parser.add_argument(
        "--auto-approve", action="store_true",
        help="skip human approval (for CI or batch runs)",
    )
    parser.add_argument(
        "--list", action="store_true", help="list discovered agents and exit",
    )
    args = parser.parse_args()

    if args.list:
        for a in load_agent_defs(args.agents_dir):
            print(f"  {a.name:<18} {a.description}  [model={a.model}]  {a.source_file}")
        return

    if not args.task:
        parser.error("task is required (or use --list)")

    run_forge(
        args.task, args.agents_dir,
        model_size=args.model,
        auto_approve=args.auto_approve,
    )


if __name__ == "__main__":
    main()
