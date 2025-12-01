"""
Image preprocessing utilities for the neural authentication system.

This module handles ONLY image preprocessing following SRP.
- Normalizes images to [-1, 1] range
- Resizes images to target size
- Adds noise and perturbations for robustness testing
"""

import numpy as np
import tensorflow as tf
from typing import Tuple, Optional
from ..config import IMAGE_SIZE, CHANNELS


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to [-1, 1] range.

    Assumes input is in [0, 1] range (common for PIL/image libraries).

    Args:
        image: Input image array

    Returns:
        Normalized image in [-1, 1]
    """
    return 2.0 * (image - 0.5)


def denormalize_image(image: np.ndarray) -> np.ndarray:
    """
    Denormalize image from [-1, 1] to [0, 1] range.

    Args:
        image: Input image in [-1, 1]

    Returns:
        Denormalized image in [0, 1]
    """
    return (image + 1.0) / 2.0


def resize_image(
    image: np.ndarray, target_size: int = IMAGE_SIZE
) -> np.ndarray:
    """
    Resize image to target size using bilinear interpolation.

    Args:
        image: Input image
        target_size: Target height and width (assumes square)

    Returns:
        Resized image
    """
    if image.shape[:2] != (target_size, target_size):
        image = tf.image.resize(image, (target_size, target_size))
        if isinstance(image, tf.Tensor):
            image = image.numpy()
    return image.astype(np.float32)


def add_gaussian_noise(
    image: np.ndarray, std: float = 0.01
) -> np.ndarray:
    """
    Add Gaussian noise to image for robustness testing.

    Args:
        image: Input image in [-1, 1]
        std: Standard deviation of Gaussian noise

    Returns:
        Noisy image clipped to [-1, 1]
    """
    noise = np.random.normal(0, std, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, -1, 1).astype(np.float32)


def add_salt_and_pepper_noise(
    image: np.ndarray, salt_pepper_ratio: float = 0.01
) -> np.ndarray:
    """
    Add salt-and-pepper noise (random black and white pixels).

    Args:
        image: Input image in [-1, 1]
        salt_pepper_ratio: Ratio of noisy pixels (0 to 1)

    Returns:
        Noisy image
    """
    noisy_image = image.copy()
    num_salt = int(np.ceil(salt_pepper_ratio * image.size / 2))
    num_pepper = int(np.ceil(salt_pepper_ratio * image.size / 2))

    # Add salt (white noise)
    coords = np.random.randint(0, image.shape[0], num_salt), np.random.randint(
        0, image.shape[1], num_salt
    )
    noisy_image[coords] = 1.0

    # Add pepper (black noise)
    coords = np.random.randint(0, image.shape[0], num_pepper), np.random.randint(
        0, image.shape[1], num_pepper
    )
    noisy_image[coords] = -1.0

    return noisy_image.astype(np.float32)


def apply_jpeg_compression(
    image: np.ndarray, quality: int = 90
) -> np.ndarray:
    """
    Simulate JPEG compression (useful for robustness testing).

    This uses TensorFlow's JPEG encoding/decoding simulation.

    Args:
        image: Input image in [-1, 1]
        quality: JPEG quality (0-100)

    Returns:
        Compressed image
    """
    # Convert from [-1, 1] to [0, 1]
    image_01 = denormalize_image(image)
    # Convert to [0, 255]
    image_255 = tf.cast(image_01 * 255, tf.uint8)
    # Encode and decode as JPEG
    encoded = tf.io.encode_jpeg(image_255, quality=quality)
    decoded = tf.io.decode_jpeg(encoded)
    # Convert back to [-1, 1]
    image_01_reconstructed = tf.cast(decoded, tf.float32) / 255.0
    image_normalized = normalize_image(image_01_reconstructed.numpy())

    return image_normalized.astype(np.float32)


def clip_image(image: np.ndarray) -> np.ndarray:
    """
    Clip image to valid range [-1, 1].

    Args:
        image: Input image

    Returns:
        Clipped image
    """
    return np.clip(image, -1, 1).astype(np.float32)


def center_crop(
    image: np.ndarray, crop_size: int
) -> np.ndarray:
    """
    Center crop image to specified size.

    Args:
        image: Input image
        crop_size: Size of crop (assumes square)

    Returns:
        Cropped image
    """
    h, w = image.shape[:2]
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    return image[start_h : start_h + crop_size, start_w : start_w + crop_size]


def random_crop(
    image: np.ndarray, crop_size: int
) -> np.ndarray:
    """
    Random crop image to specified size.

    Args:
        image: Input image
        crop_size: Size of crop (assumes square)

    Returns:
        Cropped image
    """
    h, w = image.shape[:2]
    if h < crop_size or w < crop_size:
        return image

    start_h = np.random.randint(0, h - crop_size + 1)
    start_w = np.random.randint(0, w - crop_size + 1)
    return image[start_h : start_h + crop_size, start_w : start_w + crop_size]


def preprocess_for_network(
    image: np.ndarray, target_size: int = IMAGE_SIZE
) -> np.ndarray:
    """
    Complete preprocessing pipeline for neural network input.

    Args:
        image: Input image (any range)
        target_size: Target size for resizing

    Returns:
        Preprocessed image in [-1, 1]
    """
    # Resize to target size
    image = resize_image(image, target_size)

    # Ensure in [-1, 1] range
    if image.max() > 1.0 or image.min() < -1.0:
        # Likely in [0, 255] or [0, 1]
        if image.max() > 1.0:
            image = image / 255.0
        image = normalize_image(image)

    return clip_image(image)


def postprocess_from_network(image: np.ndarray) -> np.ndarray:
    """
    Convert network output from [-1, 1] to [0, 255] for saving/display.

    Args:
        image: Network output in [-1, 1]

    Returns:
        Image in [0, 255] range as uint8
    """
    # Clip to valid range
    image = clip_image(image)
    # Convert to [0, 1]
    image = denormalize_image(image)
    # Convert to [0, 255]
    image = (image * 255).astype(np.uint8)
    return image
