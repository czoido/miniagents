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
| `--model` | `4b` | Model for debaters |
| `--judge-model` | same | Model for the judge |
| `--rounds` | `3` | Debate rounds |
| `--max-tokens` | `200` | Tokens per turn |
| `--max-tokens-judge` | `400` | Tokens for verdict |
| `--interactive` | off | Loop over topics |

## Adding experiments

```
experiments/02_name/
├── __init__.py
└── run.py          # from shared.models import MODELS, load_model
```

```bash
python -m experiments.02_name.run
```

## Links

- [Bonsai models](https://huggingface.co/collections/prism-ml/bonsai) ·
  [whitepaper](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/1-bit-bonsai-8b-whitepaper.pdf)
- [smolagents](https://github.com/huggingface/smolagents) ·
  [MLX](https://github.com/ml-explore/mlx) ·
  [BitNet](https://github.com/microsoft/BitNet)
