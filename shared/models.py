"""Shared model registry and loading utilities."""

from smolagents import MLXModel

from shared.console import console

MODELS = {
    # 1-bit (Bonsai)
    "8b": "prism-ml/Bonsai-8B-mlx-1bit",
    "4b": "prism-ml/Bonsai-4B-mlx-1bit",
    "1.7b": "prism-ml/Bonsai-1.7B-mlx-1bit",
    # 4-bit (MLX Community)
    "qwen-7b": "mlx-community/Qwen2.5-7B-Instruct-4bit",
}


def load_model(size: str = "8b") -> MLXModel:
    if size in MODELS:
        model_id = MODELS[size]
    elif "/" in size:
        model_id = size
    else:
        raise ValueError(
            f"Unknown model '{size}'. "
            f"Available: {', '.join(MODELS)} or a HuggingFace model ID"
        )
    console.print(f"  Loading [bold]{model_id}[/bold] …")
    model = MLXModel(model_id)
    console.print("  Ready.", style="green")
    console.print()
    return model
