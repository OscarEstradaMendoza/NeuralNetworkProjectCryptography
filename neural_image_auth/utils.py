"""
Utility functions for the neural authentication system.

This module implements ONLY helper functions following SRP.
Includes utilities for:
- Model saving and loading
- Visualization
- Data serialization
"""

import os
import json
import numpy as np
import tensorflow as tf
from typing import Dict, Optional, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime


def save_model(model, model_name: str, directory: str = "models") -> str:
    """
    Save a trained model to disk.

    Args:
        model: TensorFlow model to save
        model_name: Name for the model (e.g., 'alice', 'bob', 'eve')
        directory: Directory to save to

    Returns:
        Path to the saved model (Keras .keras format)
    """
    os.makedirs(directory, exist_ok=True)
    # Use native Keras v3 format with explicit extension to avoid ValueError
    model_path = os.path.join(directory, f"{model_name}.keras")
    model.save(model_path)
    print(f"Model saved to {model_path}")
    return model_path


def load_model(model_path: str):
    """
    Load a trained model from disk.

    Args:
        model_path: Path to the saved model

    Returns:
        Loaded TensorFlow model
    """
    try:
        # Import custom layers to register them with Keras
        from .models.alice import PerturbationClipLayer
        
        # Keras 3 enables safe_mode by default, which blocks Lambda layers.
        # Our models use custom layers now, and we trust these local artifacts, so we
        # explicitly disable safe_mode for deserialization.
        custom_objects = {
            'PerturbationClipLayer': PerturbationClipLayer
        }
        model = tf.keras.models.load_model(
            model_path,
            safe_mode=False,
            custom_objects=custom_objects
        )
        print(f"Model loaded from {model_path}")
        return model
    except (ValueError, TypeError, ImportError) as e:
        error_str = str(e)
        if "Lambda" in error_str or "output_shape" in error_str or "tf" in error_str:
            print("\n" + "=" * 60)
            print("ERROR: Model loading failed due to layer serialization issue.")
            print("=" * 60)
            print("\nThe saved model was created with an older layer definition")
            print("that has serialization issues with Keras 3.")
            print("\nSOLUTION: Retrain the models with the updated code:")
            print("  python -m neural_image_auth.main")
            print("\nThe Lambda layer has been replaced with a custom layer")
            print("that properly serializes and deserializes.")
            print("=" * 60 + "\n")
        raise


def save_training_config(config: Dict, filepath: str) -> None:
    """
    Save training configuration to JSON file.

    Args:
        config: Configuration dictionary
        filepath: Path to save config to
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(config, f, indent=2, default=str)
    print(f"Configuration saved to {filepath}")


def load_training_config(filepath: str) -> Dict:
    """
    Load training configuration from JSON file.

    Args:
        filepath: Path to config file

    Returns:
        Configuration dictionary
    """
    with open(filepath, "r") as f:
        config = json.load(f)
    print(f"Configuration loaded from {filepath}")
    return config


def visualize_signed_images(
    original: np.ndarray,
    signed: np.ndarray,
    difference: Optional[np.ndarray] = None,
    title: str = "Image Signing Result",
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize original, signed, and difference images side-by-side.

    Args:
        original: Original image (H, W, 3) or (H, W)
        signed: Signed image (H, W, 3) or (H, W)
        difference: Optional difference image
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
    """
    num_plots = 3 if difference is not None else 2
    fig, axes = plt.subplots(1, num_plots, figsize=figsize)

    # Original
    axes[0].imshow(np.clip(original, 0, 255).astype(np.uint8))
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Signed
    axes[1].imshow(np.clip(signed, 0, 255).astype(np.uint8))
    axes[1].set_title("Signed with Watermark")
    axes[1].axis("off")

    # Difference
    if difference is not None:
        # Normalize difference for visualization
        diff_vis = np.abs(difference)
        diff_vis = (diff_vis - diff_vis.min()) / (diff_vis.max() - diff_vis.min())
        axes[2].imshow(diff_vis, cmap="hot")
        axes[2].set_title("Perturbation Map")
        axes[2].axis("off")

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.close()


def visualize_bit_extraction(
    original_bits: np.ndarray,
    extracted_bits: np.ndarray,
    figsize: Tuple[int, int] = (14, 4),
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize original vs extracted bits as heatmaps.

    Args:
        original_bits: Original bits (message_length,)
        extracted_bits: Extracted bits (message_length,)
        figsize: Figure size
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Original bits
    axes[0].imshow(original_bits.reshape(-1, 32), cmap="RdYlGn", vmin=-1, vmax=1)
    axes[0].set_title("Original Encrypted Bits")
    axes[0].set_xlabel("Column")
    axes[0].set_ylabel("Row")

    # Extracted bits
    axes[1].imshow(extracted_bits.reshape(-1, 32), cmap="RdYlGn", vmin=-1, vmax=1)
    axes[1].set_title("Extracted Bits")
    axes[1].set_xlabel("Column")
    axes[1].set_ylabel("Row")

    # Difference
    difference = original_bits - extracted_bits
    axes[2].imshow(
        difference.reshape(-1, 32), cmap="RdYlGn", vmin=-2, vmax=2
    )
    axes[2].set_title("Difference (Original - Extracted)")
    axes[2].set_xlabel("Column")
    axes[2].set_ylabel("Row")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.close()


def save_training_history(
    history: Dict,
    filepath: str = "logs/training_history.json",
) -> None:
    """
    Save training history to JSON file.

    Args:
        history: Dictionary with loss/metric values per epoch
        filepath: Path to save to
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
    history_serializable = {}
    for key, values in history.items():
        if isinstance(values, np.ndarray):
            history_serializable[key] = values.tolist()
        elif isinstance(values, list):
            history_serializable[key] = [
                float(v) if isinstance(v, (np.floating, np.ndarray)) else v
                for v in values
            ]
        else:
            history_serializable[key] = values

    with open(filepath, "w") as f:
        json.dump(history_serializable, f, indent=2)

    print(f"Training history saved to {filepath}")


def plot_training_history(
    history: Dict,
    metrics: list = None,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot training history.

    Args:
        history: Dictionary with loss/metric values per epoch
        metrics: List of metric names to plot (default: plot all)
        figsize: Figure size
        save_path: Optional path to save figure
    """
    if metrics is None:
        metrics = list(history.keys())

    num_metrics = len(metrics)
    fig, axes = plt.subplots(
        (num_metrics + 1) // 2, 2 if num_metrics > 1 else 1, figsize=figsize
    )

    if num_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, metric in enumerate(metrics):
        if metric in history:
            values = history[metric]
            axes[i].plot(values)
            axes[i].set_title(metric)
            axes[i].set_xlabel("Epoch")
            axes[i].set_ylabel("Value")
            axes[i].grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(num_metrics, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.close()


def calculate_model_size(model) -> Dict[str, float]:
    """
    Calculate model size statistics.

    Args:
        model: TensorFlow model

    Returns:
        Dictionary with size information
    """
    total_params = model.count_params()
    trainable_params = sum(
        tf.size(w).numpy() for w in model.trainable_weights
    )
    non_trainable_params = total_params - trainable_params

    return {
        "total_parameters": int(total_params),
        "trainable_parameters": int(trainable_params),
        "non_trainable_parameters": int(non_trainable_params),
        "total_mb": round(total_params * 4 / (1024 * 1024), 2),  # Assuming float32
    }


def print_model_summary(model, model_name: str = "Model") -> None:
    """
    Print model architecture and parameters summary.

    Args:
        model: TensorFlow model
        model_name: Name to display
    """
    print(f"\n{'='*60}")
    print(f"{model_name} Summary")
    print(f"{'='*60}")
    model.summary()

    sizes = calculate_model_size(model)
    print(f"\nModel Size Information:")
    print(f"  Total Parameters:     {sizes['total_parameters']:,}")
    print(f"  Trainable Parameters: {sizes['trainable_parameters']:,}")
    print(f"  Non-trainable Params: {sizes['non_trainable_parameters']:,}")
    print(f"  Total Size (MB):      {sizes['total_mb']}")
    print(f"{'='*60}\n")


def get_timestamp() -> str:
    """Get current timestamp as string for logging."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def create_log_directory(base_dir: str = "logs") -> str:
    """
    Create a timestamped log directory.

    Args:
        base_dir: Base directory for logs

    Returns:
        Path to created directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(base_dir, f"train_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir
