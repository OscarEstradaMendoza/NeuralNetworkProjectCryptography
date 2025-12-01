"""
Synthetic image data generation for training the neural authentication system.

This module handles ONLY synthetic image generation following SRP.
- Generates random images with various patterns
- Creates batches of images for training
- Provides different image generation strategies
"""

import numpy as np
import tensorflow as tf
from typing import Tuple
from ..config import IMAGE_SIZE, CHANNELS, BATCH_SIZE


class ImageGenerator:
    """Generate synthetic training images."""

    def __init__(self, image_size: int = IMAGE_SIZE, channels: int = CHANNELS):
        """
        Initialize image generator.

        Args:
            image_size: Size of square images (default 64×64)
            channels: Number of channels (default 3 for RGB)
        """
        self.image_size = image_size
        self.channels = channels

    def generate_random_images(self, batch_size: int = BATCH_SIZE) -> np.ndarray:
        """
        Generate random noise images.

        Args:
            batch_size: Number of images to generate

        Returns:
            Array of shape (batch_size, image_size, image_size, channels)
            with values in [-1, 1]
        """
        images = np.random.uniform(
            -1, 1, (batch_size, self.image_size, self.image_size, self.channels)
        )
        return images.astype(np.float32)

    def generate_pattern_images(self, batch_size: int = BATCH_SIZE) -> np.ndarray:
        """
        Generate images with geometric patterns (gradients, checkerboards, etc.).

        Args:
            batch_size: Number of images to generate

        Returns:
            Array of shape (batch_size, image_size, image_size, channels)
            with values in [-1, 1]
        """
        images = []

        for _ in range(batch_size):
            img = np.zeros((self.image_size, self.image_size, self.channels))

            pattern_type = np.random.randint(0, 4)

            if pattern_type == 0:
                # Gradient pattern
                for i in range(self.image_size):
                    img[i, :, :] = 2 * (i / self.image_size) - 1

            elif pattern_type == 1:
                # Checkerboard pattern
                square_size = self.image_size // 4
                for i in range(0, self.image_size, square_size):
                    for j in range(0, self.image_size, square_size):
                        if ((i // square_size) + (j // square_size)) % 2 == 0:
                            img[i : i + square_size, j : j + square_size, :] = 0.8
                        else:
                            img[i : i + square_size, j : j + square_size, :] = -0.8

            elif pattern_type == 2:
                # Circular pattern
                center = self.image_size // 2
                for i in range(self.image_size):
                    for j in range(self.image_size):
                        dist = np.sqrt((i - center) ** 2 + (j - center) ** 2)
                        img[i, j, :] = 2 * (dist / (center * 1.5)) - 1

            else:
                # Diagonal stripes
                stripe_width = self.image_size // 4
                for i in range(self.image_size):
                    for j in range(self.image_size):
                        if ((i + j) // stripe_width) % 2 == 0:
                            img[i, j, :] = 0.8
                        else:
                            img[i, j, :] = -0.8

            images.append(np.clip(img, -1, 1))

        return np.array(images, dtype=np.float32)

    def generate_mixed_images(self, batch_size: int = BATCH_SIZE) -> np.ndarray:
        """
        Generate images by mixing random and pattern images.

        Args:
            batch_size: Number of images to generate

        Returns:
            Array of shape (batch_size, image_size, image_size, channels)
        """
        random_images = self.generate_random_images(batch_size // 2)
        pattern_images = self.generate_pattern_images(batch_size - batch_size // 2)
        return np.vstack([random_images, pattern_images])

    def generate_gaussian_images(self, batch_size: int = BATCH_SIZE) -> np.ndarray:
        """
        Generate images with Gaussian-like distributions.

        Args:
            batch_size: Number of images to generate

        Returns:
            Array of shape (batch_size, image_size, image_size, channels)
        """
        images = np.random.normal(
            0, 0.5, (batch_size, self.image_size, self.image_size, self.channels)
        )
        images = np.clip(images, -1, 1)
        return images.astype(np.float32)


class DataPipeline:
    """Create training data pipeline."""

    def __init__(
        self,
        batch_size: int = BATCH_SIZE,
        image_size: int = IMAGE_SIZE,
        channels: int = CHANNELS,
    ):
        """
        Initialize data pipeline.

        Args:
            batch_size: Batch size for training
            image_size: Size of images
            channels: Number of channels
        """
        self.batch_size = batch_size
        self.image_generator = ImageGenerator(image_size, channels)

    def get_training_batch(self) -> tf.Tensor:
        """
        Get a batch of training images.

        Returns:
            TensorFlow tensor of shape (batch_size, image_size, image_size, channels)
        """
        images = self.image_generator.generate_mixed_images(self.batch_size)
        return tf.constant(images)

    def get_validation_batch(self) -> tf.Tensor:
        """Get a batch of validation images."""
        images = self.image_generator.generate_random_images(self.batch_size)
        return tf.constant(images)

    def get_test_batch(self) -> tf.Tensor:
        """Get a batch of test images."""
        images = self.image_generator.generate_pattern_images(self.batch_size)
        return tf.constant(images)
