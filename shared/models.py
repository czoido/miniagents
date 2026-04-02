"""Shared model registry and loading utilities."""

from smolagents import MLXModel

MODELS = {
    "8b": "prism-ml/Bonsai-8B-mlx-1bit",
    "4b": "prism-ml/Bonsai-4B-mlx-1bit",
    "1.7b": "prism-ml/Bonsai-1.7B-mlx-1bit",
}


def load_model(size: str = "4b") -> MLXModel:
    model_id = MODELS[size]
    print(f"  Loading {model_id} ...")
    model = MLXModel(model_id)
    print(f"  Ready.\n")
    return model
