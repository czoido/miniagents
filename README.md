# miniagents

Multi-agent coordination experiments with 1-bit LLMs
([Bonsai](https://huggingface.co/collections/prism-ml/bonsai)) on Apple Silicon
using [smolagents](https://github.com/huggingface/smolagents) and
[MLX](https://github.com/ml-explore/mlx).

## Setup

Requires macOS with Apple Silicon.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Models download automatically on first run.

## Models

| Model | RAM |
|-------|-----|
| `1.7b` — `prism-ml/Bonsai-1.7B-mlx-1bit` | ~300 MB |
| `4b` — `prism-ml/Bonsai-4B-mlx-1bit` | ~600 MB |
| `8b` — `prism-ml/Bonsai-8B-mlx-1bit` | ~1.2 GB |

## Experiments

Run from repo root.

### 01 — Adversarial Debate

An advocate and a critic debate a proposition across multiple rounds. A judge
reads the transcript and delivers a verdict.

```bash
python -m experiments.01_debate.run "Should programming languages have garbage collection?"
python -m experiments.01_debate.run "Is open source AI safer than closed source?" --rounds 5
python -m experiments.01_debate.run "Will LLMs replace programmers?" --judge-model 8b
python -m experiments.01_debate.run --interactive
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `8b` | Model for debaters |
| `--judge-model` | same | Model for the judge |
| `--rounds` | `3` | Debate rounds |
| `--max-tokens` | `200` | Tokens per turn |
| `--max-tokens-judge` | `400` | Tokens for verdict |
| `--interactive` | off | Loop over topics |

### 02 — HN Digest

Fetches top Hacker News stories, extracts article content, summarizes each
with a dedicated agent, and a curator produces a ranked digest.

```bash
python -m experiments.02_hn_digest.run
python -m experiments.02_hn_digest.run --stories 10
python -m experiments.02_hn_digest.run --stories 3 --model 1.7b
```

| Flag | Default | Description |
|------|---------|-------------|
| `--stories` | `5` | Stories to process |
| `--model` | `8b` | Model size |
| `--max-article-chars` | `3000` | Max chars extracted per article |

### 03 — Exquisite Corpse

Surrealist collaborative storytelling. Multiple agents with distinct voices
(poet, noir, sci-fi, absurdist) take turns writing, but each only sees the
last few sentences from the previous turn. Less overlap = more surreal.

```bash
python -m experiments.03_exquisite_corpse.run
python -m experiments.03_exquisite_corpse.run --seed "The door opened slowly"
python -m experiments.03_exquisite_corpse.run --turns 10 --overlap 1
python -m experiments.03_exquisite_corpse.run --overlap 1 --model 8b
```

| Flag | Default | Description |
|------|---------|-------------|
| `--seed` | random | Opening sentence |
| `--turns` | `6` | Number of turns |
| `--overlap` | `2` | Sentences visible to next agent |
| `--model` | `8b` | Model size |
| `--max-tokens` | `150` | Tokens per turn |

### 04 — Agent Forge

Dynamic agent discovery from markdown definitions. Agent personalities and
capabilities are defined as `.md` files in `agents/`. Uses the full smolagents
`CodeAgent` + `ManagedAgent` pipeline: the coordinator writes Python to call
sub-agents, but every action requires human approval before execution.

Includes a `web_researcher` agent type that searches the web via DuckDuckGo,
fetches pages, and synthesizes answers — no browser required.

```bash
python -m experiments.04_agent_forge.run "Write a poem about the sea"
python -m experiments.04_agent_forge.run "Analyze the pros and cons of AI"
python -m experiments.04_agent_forge.run --auto-approve "Quick test"
python -m experiments.04_agent_forge.run --list
```

| Flag | Default | Description |
|------|---------|-------------|
| `--agents-dir` | `agents/` | Directory with `.md` agent definitions |
| `--model` | `8b` | Model size for coordinator |
| `--auto-approve` | off | Skip human approval (for CI) |
| `--list` | off | List discovered agents and exit |

Each sub-agent can override the model in its own markdown definition:

```markdown
---
name: poet
description: Writes creative literary prose
model: 8b
temp: 0.9
max_tokens: 300
---

You are a poet. Write vivid, metaphorical prose...
```

| Frontmatter field | Default | Description |
|-------------------|---------|-------------|
| `name` | filename stem | Agent identifier |
| `description` | — | What the agent does (shown to coordinator) |
| `type` | `text` | Agent type: `text` or `web_researcher` |
| `model` | `8b` | Model size: `8b`, `4b`, or `1.7b` |
| `temp` | `0.7` | Sampling temperature |
| `max_tokens` | `300` | Max generation tokens |
| `max_sources` | `3` | Max pages to fetch (web_researcher only) |

## Adding experiments

```
experiments/NN_name/
├── __init__.py
└── run.py          # from shared.models import MODELS, load_model
```

```bash
python -m experiments.NN_name.run
```

## Links

- [Bonsai models](https://huggingface.co/collections/prism-ml/bonsai) ·
  [whitepaper](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/1-bit-bonsai-8b-whitepaper.pdf)
- [smolagents](https://github.com/huggingface/smolagents) ·
  [MLX](https://github.com/ml-explore/mlx) ·
  [BitNet](https://github.com/microsoft/BitNet)
