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
from smolagents import CodeAgent, MLXModel

from shared.models import MODELS

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

W = 64


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
            print(f"\n{'─' * W}")
            print(f"  \033[33m⚠  The agent wants to execute:\033[0m\n")
            for line in code_action.splitlines():
                print(f"    \033[36m{line}\033[0m")
            print(f"\n{'─' * W}")

            while True:
                answer = input("  Approve? [y]es / [n]o / [q]uit: ").strip().lower()
                if answer in ("y", "yes", ""):
                    break
                if answer in ("n", "no"):
                    raise RuntimeError("Denied by user. Try a different approach.")
                if answer in ("q", "quit"):
                    print("\n  Aborted.")
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
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self._instructions}]},
            {"role": "user", "content": [{"type": "text", "text": task}]},
        ]
        response = self.model(messages, max_tokens=self._max_tokens)
        text = response.content
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
                    print(f"    \033[35m🔍 {self.name}\033[0m retry {attempt + 1}/{max_retries} (empty results)...")
                    time.sleep(1)
            except Exception as e:
                if attempt == max_retries - 1:
                    return f"Search failed after {max_retries} attempts: {e}"
                time.sleep(1)
        return result or "No search results found."

    def __call__(self, task: str, **kwargs) -> str:
        self._ensure_tools()
        print(f"    \033[35m🔍 {self.name}\033[0m searching: {task[:80]}...")

        search_result = self._search_with_retry(task)

        urls = self._extract_urls(search_result)
        page_excerpts: list[str] = []
        max_chars_per_page = 2000

        for url in urls[: self._max_sources]:
            print(f"    \033[35m🔍 {self.name}\033[0m fetching: {url[:60]}...")
            try:
                content = self._visit(url)
                page_excerpts.append(
                    f"### Source: {url}\n{content[:max_chars_per_page]}"
                )
            except Exception as e:
                print(f"    \033[35m🔍 {self.name}\033[0m \033[31mfetch failed:\033[0m {e}")
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
        print(f"  Loading {model_id} ...")
        models[size] = MLXModel(model_id)
    print(f"  Ready.\n")
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
        print("  No agent definitions found in", agents_dir)
        return

    print(f"\n{'═' * W}")
    print(f"  AGENT FORGE — {len(defs)} agents discovered")
    print(f"{'═' * W}\n")

    for d in defs:
        model_tag = f" [{d.model}]" if d.model != model_size else ""
        print(f"    \033[36m●\033[0m {d.name:<18} {d.description}{model_tag}")
    print(f"\n  \033[1mTask:\033[0m \"{task}\"")

    if not auto_approve:
        print(f"\n  \033[33m⚠  Human approval is ON\033[0m — you will review "
              f"each action before execution.")
    print()

    models = _load_models(defs, model_size)
    managed = build_managed_agents(defs, models)

    agent_list = ", ".join(f"`{d.name}`" for d in defs)
    coordinator_instructions = (
        "You are a coordinator that delegates work to specialized agents.\n\n"
        f"AVAILABLE AGENTS: {agent_list}\n"
        "You have NO other tools — no web_search, no wikipedia_search, no file access. "
        "Do NOT call any function not listed above.\n\n"
        "RULES:\n"
        "1. Call an agent: result = agent_name(task=\"description\")\n"
        "2. Store the result in a variable, then pass it to the next agent if needed.\n"
        "3. ALWAYS use `final_answer(result)` inside a <code> block to return your answer.\n"
        "4. Do NOT write 'Final Answer:' as plain text. It MUST be code: final_answer(result)\n"
        "5. If an agent already gave you a good answer, just pass it through:\n\n"
        "Example with one agent:\n"
        "<code>\n"
        "result = web_researcher(task=\"What is X?\")\n"
        "final_answer(result)\n"
        "</code>\n\n"
        "Example chaining two agents:\n"
        "<code>\n"
        "research = web_researcher(task=\"Find info about X\")\n"
        "summary = summarizer(task=research)\n"
        "final_answer(summary)\n"
        "</code>"
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

    print(f"\n{'═' * W}")
    print(f"  RESULT")
    print(f"{'═' * W}\n")
    if result:
        prefix = " " * 4
        for line in str(result).splitlines():
            if line.strip():
                print(textwrap.fill(
                    line, width=W, initial_indent=prefix,
                    subsequent_indent=prefix,
                ))
            else:
                print()
    print(f"\n{'═' * W}\n")


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
        "--model", choices=MODELS, default="8b",
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
