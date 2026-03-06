"""
Shared MLFlow helpers used across all training scripts.

Loads DagsHub credentials from .env and sets up the tracking URI.
Call setup_mlflow() at the top of any training or evaluation script.
"""

import os
from pathlib import Path


def setup_mlflow() -> str:
    """
    Configure MLFlow to log to DagsHub (or localhost fallback).

    Reads from environment variables (set in .env):
        MLFLOW_TRACKING_URI
        MLFLOW_TRACKING_USERNAME
        MLFLOW_TRACKING_PASSWORD

    Returns the active tracking URI.
    """
    # Load .env if present
    env_file = Path(".env")
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(env_file, override=False)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

    import mlflow
    mlflow.set_tracking_uri(tracking_uri)

    print(f"MLFlow tracking URI: {tracking_uri}")
    return tracking_uri


def log_system_info() -> None:
    """Log GPU, CUDA, and package versions as MLFlow tags."""
    import mlflow
    import torch

    mlflow.set_tag("python_version", __import__("sys").version.split()[0])
    mlflow.set_tag("torch_version", torch.__version__)
    mlflow.set_tag("cuda_version", torch.version.cuda or "N/A")

    if torch.cuda.is_available():
        mlflow.set_tag("gpu_name", torch.cuda.get_device_name(0))
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        mlflow.set_tag("gpu_vram_gb", f"{vram_gb:.1f}")
