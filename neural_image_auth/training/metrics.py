"""
Evaluation metrics for monitoring training progress.

This module implements ONLY evaluation metric definitions following SRP.
Metrics include:
- Bit error rate
- Peak signal-to-noise ratio (PSNR)
- Authentication accuracy
- Message extraction accuracy
"""

import tensorflow as tf
import numpy as np
from typing import Tuple


def calculate_ber(
    original_bits: np.ndarray, extracted_bits: np.ndarray
) -> float:
    """
    Calculate Bit Error Rate (BER).

    BER = (number of incorrectly extracted bits) / (total bits)

    Args:
        original_bits: Ground truth bits of shape (batch_size, message_length)
                       with values in [-1, 1]
        extracted_bits: Extracted bits of shape (batch_size, message_length)
                        with values in [-1, 1]

    Returns:
        BER as float in [0, 1]
    """
    # Threshold at 0 for discrete comparison
    original_hard = np.sign(original_bits)
    extracted_hard = np.sign(extracted_bits)

    # Count errors
    errors = np.sum(original_hard != extracted_hard)
    total = original_bits.size

    ber = errors / total
    return float(ber)


def calculate_psnr(
    original: np.ndarray, reconstructed: np.ndarray
) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).

    PSNR is commonly used to measure image quality.
    Higher PSNR indicates better quality.

    Args:
        original: Original image in [-1, 1]
        reconstructed: Reconstructed/perturbed image in [-1, 1]

    Returns:
        PSNR in dB (decibels)
    """
    # MSE
    mse = np.mean((original - reconstructed) ** 2)

    if mse == 0:
        return float("inf")  # Perfect reconstruction

    # Signal range in [-1, 1] is 2
    signal_range = 2.0
    psnr = 20 * np.log10(signal_range / np.sqrt(mse))

    return float(psnr)


def calculate_authentication_accuracy(
    predictions: np.ndarray, labels: np.ndarray, threshold: float = 0.5
) -> float:
    """
    Calculate authentication classification accuracy.

    Args:
        predictions: Bob's authenticity predictions in [0, 1]
        labels: Ground truth labels (1 for authentic, 0 for non-authentic)
        threshold: Classification threshold (default 0.5)

    Returns:
        Accuracy as float in [0, 1]
    """
    predicted_labels = (predictions > threshold).astype(int)
    accuracy = np.mean(predicted_labels == labels)
    return float(accuracy)


def calculate_message_extraction_accuracy(
    original_bits: np.ndarray, extracted_bits: np.ndarray
) -> float:
    """
    Calculate message extraction accuracy (complement of BER).

    Args:
        original_bits: Ground truth bits of shape (..., message_length)
        extracted_bits: Extracted bits of shape (..., message_length)

    Returns:
        Accuracy as float in [0, 1]
    """
    ber = calculate_ber(original_bits, extracted_bits)
    return 1.0 - ber


def calculate_metrics_batch(
    original_images: np.ndarray,
    perturbed_images: np.ndarray,
    original_bits: np.ndarray,
    extracted_bits: np.ndarray,
    auth_predictions: np.ndarray,
    auth_labels: np.ndarray,
) -> dict:
    """
    Calculate all metrics for a batch of data.

    Args:
        original_images: Original images
        perturbed_images: Perturbed/signed images
        original_bits: Ground truth message bits
        extracted_bits: Bob's extracted bits
        auth_predictions: Bob's authenticity predictions
        auth_labels: Ground truth authenticity labels

    Returns:
        Dictionary with all metrics
    """
    metrics = {}

    # Image quality metrics
    psnr_values = []
    for orig, perturbed in zip(original_images, perturbed_images):
        psnr = calculate_psnr(orig, perturbed)
        if not np.isinf(psnr):  # Exclude infinite values
            psnr_values.append(psnr)

    metrics["psnr_mean"] = float(np.mean(psnr_values)) if psnr_values else 0.0
    metrics["psnr_min"] = float(np.min(psnr_values)) if psnr_values else 0.0

    # Message extraction metrics
    metrics["ber"] = calculate_ber(original_bits, extracted_bits)
    metrics["message_accuracy"] = calculate_message_extraction_accuracy(
        original_bits, extracted_bits
    )

    # Authentication metrics
    metrics["auth_accuracy"] = calculate_authentication_accuracy(
        auth_predictions, auth_labels
    )

    return metrics


def calculate_sensitivity(
    bob_model,
    original_images: np.ndarray,
    perturbed_by_alice: np.ndarray,
) -> float:
    """
    Calculate sensitivity of Bob to authentic images.

    Sensitivity = True Positive Rate = how many authentic images does Bob correctly accept.

    Args:
        bob_model: Trained Bob network
        original_images: Original images (for negative samples)
        perturbed_by_alice: Images signed by Alice (for positive samples)

    Returns:
        Sensitivity as float in [0, 1]
    """
    # Predictions on Alice-signed images
    _, auth_pred_positive = bob_model(perturbed_by_alice, training=False)
    tp = np.sum((auth_pred_positive.numpy() > 0.5).astype(int))
    total_positive = len(perturbed_by_alice)

    sensitivity = tp / total_positive if total_positive > 0 else 0.0
    return float(sensitivity)


def calculate_specificity(
    bob_model,
    original_images: np.ndarray,
) -> float:
    """
    Calculate specificity of Bob to non-authentic images.

    Specificity = True Negative Rate = how many non-authentic images does Bob correctly reject.

    Args:
        bob_model: Trained Bob network
        original_images: Original unsigned images (negative samples)

    Returns:
        Specificity as float in [0, 1]
    """
    # Predictions on unsigned images
    _, auth_pred_negative = bob_model(original_images, training=False)
    tn = np.sum((auth_pred_negative.numpy() <= 0.5).astype(int))
    total_negative = len(original_images)

    specificity = tn / total_negative if total_negative > 0 else 0.0
    return float(specificity)
