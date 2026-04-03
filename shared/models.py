"""Shared model registry and loading utilities."""

from smolagents import MLXModel

from shared.console import console

MODELS = {
    "8b": "prism-ml/Bonsai-8B-mlx-1bit",
    "4b": "prism-ml/Bonsai-4B-mlx-1bit",
    "1.7b": "prism-ml/Bonsai-1.7B-mlx-1bit",
}


def load_model(size: str = "8b") -> MLXModel:
    model_id = MODELS[size]
    console.print(f"  Loading [bold]{model_id}[/bold] …")
    model = MLXModel(model_id)
    console.print("  Ready.", style="green")
    console.print()
    return model
