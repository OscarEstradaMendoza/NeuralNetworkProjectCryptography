"""
Loss functions for training the neural authentication system.

This module implements ONLY loss function definitions following SRP.
Includes losses for:
- Reconstruction (perturbed image close to original)
- Message extraction (correct bit recovery)
- Authentication classification (authentic vs non-authentic)
- Imperceptibility (bounded perturbation)
"""

import tensorflow as tf
from typing import Dict, Tuple


def reconstruction_loss(original: tf.Tensor, perturbed: tf.Tensor) -> tf.Tensor:
    """
    Reconstruction loss - MSE between original and perturbed images.

    Ensures the perturbed image stays close to the original (imperceptibility).

    Args:
        original: Original images of shape (batch_size, height, width, channels)
        perturbed: Perturbed images of shape (batch_size, height, width, channels)

    Returns:
        Scalar loss value (mean squared error)
    """
    return tf.reduce_mean(tf.square(original - perturbed))


def message_extraction_loss(
    original_bits: tf.Tensor, extracted_bits: tf.Tensor
) -> tf.Tensor:
    """
    Message extraction loss - MSE for bit recovery.

    Measures how well Bob can extract the message bits.
    Bits are in continuous {-1, 1} space, so MSE is appropriate.

    Args:
        original_bits: Ground truth bits of shape (batch_size, message_length)
                       with values in [-1, 1]
        extracted_bits: Extracted bits of shape (batch_size, message_length)
                        with values in [-1, 1]

    Returns:
        Scalar loss value (mean squared error)
    """
    return tf.reduce_mean(tf.square(original_bits - extracted_bits))


def bit_accuracy(
    original_bits: tf.Tensor, extracted_bits: tf.Tensor
) -> tf.Tensor:
    """
    Calculate percentage of correctly extracted bits.

    Bits are thresholded at 0: negative → -1, positive → +1.
    This gives a discrete accuracy metric for monitoring.

    Args:
        original_bits: Ground truth bits of shape (batch_size, message_length)
        extracted_bits: Extracted bits of shape (batch_size, message_length)

    Returns:
        Accuracy as float in [0, 1]
    """
    # Threshold at 0
    original_hard = tf.sign(original_bits)
    extracted_hard = tf.sign(extracted_bits)

    # Compare
    correct = tf.equal(original_hard, extracted_hard)

    # Return accuracy
    return tf.reduce_mean(tf.cast(correct, tf.float32))


def authentication_loss(
    predictions: tf.Tensor, labels: tf.Tensor
) -> tf.Tensor:
    """
    Authentication loss - Binary cross-entropy for authenticity classification.

    Measures how well Bob can distinguish authentic vs non-authentic images.

    Args:
        predictions: Bob's authenticity predictions of shape (batch_size, 1)
                     with values in [0, 1]
        labels: Ground truth labels of shape (batch_size, 1)
                with values 1 (authentic) or 0 (non-authentic)

    Returns:
        Scalar binary cross-entropy loss
    """
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    return bce(labels, predictions)


def imperceptibility_loss(
    original: tf.Tensor, perturbed: tf.Tensor, bound: float = 0.1
) -> tf.Tensor:
    """
    Imperceptibility loss - Penalize perturbations exceeding the bound.

    Enforces L∞ norm constraint on perturbations (max absolute difference).

    Args:
        original: Original images of shape (batch_size, height, width, channels)
        perturbed: Perturbed images of shape (batch_size, height, width, channels)
        bound: Maximum allowed L∞ norm of perturbation

    Returns:
        Scalar loss value (violation of bound)
    """
    # Calculate perturbation
    perturbation = tf.abs(perturbed - original)

    # L∞ norm per image (max over all pixels and channels)
    max_perturbation = tf.reduce_max(perturbation, axis=[1, 2, 3])

    # Penalize violations of bound
    violation = tf.maximum(0.0, max_perturbation - bound)

    return tf.reduce_mean(violation)


def alice_bob_combined_loss(
    original_image: tf.Tensor,
    perturbed_image: tf.Tensor,
    original_bits: tf.Tensor,
    extracted_bits: tf.Tensor,
    authenticity_pred: tf.Tensor,
    lambdas: Dict[str, float],
) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
    """
    Combined loss for training Alice and Bob together.

    Combines multiple objectives:
    1. Reconstruction: Keep perturbed image close to original
    2. Message extraction: Bob should correctly extract message
    3. Authentication: Bob should classify as authentic
    4. Imperceptibility: Bound perturbation magnitude

    Args:
        original_image: Original images (batch_size, height, width, channels)
        perturbed_image: Alice's output (batch_size, height, width, channels)
        original_bits: Ground truth encrypted message (batch_size, message_length)
        extracted_bits: Bob's extracted bits (batch_size, message_length)
        authenticity_pred: Bob's authenticity prediction (batch_size, 1)
        lambdas: Dictionary with keys:
            - 'reconstruction': weight for reconstruction loss
            - 'message': weight for message extraction loss
            - 'authentication': weight for authenticity loss
            - 'imperceptibility': weight for imperceptibility loss

    Returns:
        Tuple of (total_loss, loss_dict):
            - total_loss: Weighted sum of all losses
            - loss_dict: Dictionary of individual losses for logging
    """
    # Calculate individual losses
    l_recon = reconstruction_loss(original_image, perturbed_image)
    l_message = message_extraction_loss(original_bits, extracted_bits)

    # For authenticity loss, authentic images should be labeled 1
    l_auth = authentication_loss(authenticity_pred, tf.ones_like(authenticity_pred))

    l_imper = imperceptibility_loss(original_image, perturbed_image)

    # Weighted combination
    total = (
        lambdas["reconstruction"] * l_recon
        + lambdas["message"] * l_message
        + lambdas["authentication"] * l_auth
        + lambdas["imperceptibility"] * l_imper
    )

    # Return total loss and individual components for logging
    loss_dict = {
        "reconstruction": l_recon,
        "message": l_message,
        "authentication": l_auth,
        "imperceptibility": l_imper,
    }

    return total, loss_dict


def eve_loss(
    bob_auth_pred_on_forged: tf.Tensor,
    original_bits: tf.Tensor,
    eve_extracted_bits: tf.Tensor,
) -> tf.Tensor:
    """
    Eve's loss - attempts to both fool Bob and extract the message.

    Eve tries to:
    1. Create forged images that Bob classifies as authentic
    2. Extract the message bits without the AES key

    Args:
        bob_auth_pred_on_forged: Bob's authenticity prediction on Eve's forged images
                                 (batch_size, 1)
        original_bits: Ground truth encrypted message bits (batch_size, message_length)
        eve_extracted_bits: Eve's extracted bits (batch_size, message_length)

    Returns:
        Scalar loss value representing Eve's objective
    """
    # Eve wants to fool Bob (make forged images look authentic)
    l_fool_bob = authentication_loss(
        bob_auth_pred_on_forged, tf.ones_like(bob_auth_pred_on_forged)
    )

    # Eve wants to extract the message bits (secondary objective)
    l_extract = message_extraction_loss(original_bits, eve_extracted_bits)

    # Combine with weight on extraction (0.5) lower than fooling Bob
    return l_fool_bob + 0.5 * l_extract
