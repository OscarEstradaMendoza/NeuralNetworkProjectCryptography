"""
Training loop orchestration for the neural authentication system.

This module handles ONLY training loop logic following SRP.
Implements adversarial training with phases:
1. Train Alice + Bob together
2. Train Bob classifier
3. Train Eve adversary
4. Harden Bob against Eve
"""

import tensorflow as tf
import numpy as np
from typing import Dict, Tuple, Optional
from .losses import (
    alice_bob_combined_loss,
    eve_loss,
    bit_accuracy,
)
from .metrics import calculate_metrics_batch
from ..crypto.aes_cipher import AESCipher
from ..config import (
    LEARNING_RATE,
    BATCH_SIZE,
    MESSAGE_LENGTH,
    LAMBDAS,
)


class AdversarialTrainer:
    """Orchestrate adversarial training of Alice, Bob, and Eve."""

    def __init__(
        self,
        alice,
        bob,
        eve,
        aes_key: Optional[bytes] = None,
        learning_rate: float = LEARNING_RATE,
    ):
        """
        Initialize adversarial trainer.

        Args:
            alice: Alice network
            bob: Bob network
            eve: Eve network
            aes_key: AES key for encryption. If None, generates random key.
            learning_rate: Learning rate for optimizers
        """
        self.alice = alice
        self.bob = bob
        self.eve = eve

        # AES cipher for message encryption
        self.aes_cipher = AESCipher(aes_key)

        # Optimizers
        self.optimizer_ab = tf.keras.optimizers.Adam(learning_rate)
        self.optimizer_eve = tf.keras.optimizers.Adam(learning_rate)

    def train_step_alice_bob(
        self,
        images: tf.Tensor,
        message_bits: tf.Tensor,
    ) -> Tuple[tf.Tensor, Dict, tf.Tensor]:
        """
        Train Alice and Bob together in one gradient step.

        Alice embeds the encrypted message into images.
        Bob extracts the message and classifies as authentic.

        Args:
            images: Batch of images (batch_size, 64, 64, 3)
            message_bits: Encrypted message bits (batch_size, message_length)

        Returns:
            Tuple of (total_loss, loss_dict, bit_accuracy):
                - total_loss: Scalar loss value
                - loss_dict: Dictionary of individual loss components
                - bit_accuracy: Message extraction accuracy
        """
        with tf.GradientTape() as tape:
            # Alice embeds message
            perturbed = self.alice([images, message_bits], training=True)

            # Bob extracts and classifies
            extracted_bits, auth_pred = self.bob(perturbed, training=True)

            # Combined loss
            total_loss, loss_dict = alice_bob_combined_loss(
                images, perturbed, message_bits, extracted_bits, auth_pred, LAMBDAS
            )

        # Update both networks
        trainable_vars = self.alice.trainable_variables + self.bob.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer_ab.apply_gradients(zip(gradients, trainable_vars))

        # Calculate bit accuracy
        acc = bit_accuracy(message_bits, extracted_bits)

        return total_loss, loss_dict, acc

    def train_step_bob_classifier(
        self,
        images: tf.Tensor,
        message_bits: tf.Tensor,
    ) -> tf.Tensor:
        """
        Train Bob's authenticity classifier on mixed batch.

        Creates a batch mixing:
        - Authentic images (signed by Alice)
        - Non-authentic images (unsigned originals)

        Args:
            images: Batch of images
            message_bits: Encrypted message bits

        Returns:
            Classification loss
        """
        # Create authentic images (signed by Alice)
        authentic = self.alice([images, message_bits], training=False)

        # Create non-authentic (original images without signature)
        non_authentic = images

        # Mixed batch
        mixed = tf.concat([authentic, non_authentic], axis=0)
        labels = tf.concat(
            [tf.ones((tf.shape(images)[0], 1)), tf.zeros((tf.shape(images)[0], 1))],
            axis=0,
        )

        with tf.GradientTape() as tape:
            _, auth_pred = self.bob(mixed, training=True)
            from .losses import authentication_loss

            loss = authentication_loss(auth_pred, labels)

        gradients = tape.gradient(loss, self.bob.trainable_variables)
        self.optimizer_ab.apply_gradients(zip(gradients, self.bob.trainable_variables))

        return loss

    def train_step_eve(
        self,
        images: tf.Tensor,
        message_bits: tf.Tensor,
    ) -> tf.Tensor:
        """
        Train Eve to forge signatures and extract messages.

        Eve tries to:
        1. Create forged images that Bob classifies as authentic
        2. Extract message bits without the AES key

        Args:
            images: Batch of images
            message_bits: Encrypted message bits

        Returns:
            Eve's loss value
        """
        with tf.GradientTape() as tape:
            # Eve tries to forge
            forged = self.eve([images, message_bits], training=True)

            # Bob evaluates forgery (frozen - not trained here)
            eve_extracted, bob_auth_pred = self.bob(forged, training=False)

            # Eve's loss
            loss = eve_loss(bob_auth_pred, message_bits, eve_extracted)

        gradients = tape.gradient(loss, self.eve.trainable_variables)
        self.optimizer_eve.apply_gradients(zip(gradients, self.eve.trainable_variables))

        return loss

    def train_step_harden_bob(
        self,
        images: tf.Tensor,
        message_bits: tf.Tensor,
    ) -> tf.Tensor:
        """
        Train Bob to reject Eve's forged images.

        After Eve is trained, we train Bob to better detect Eve's forgeries.

        Args:
            images: Batch of images
            message_bits: Encrypted message bits

        Returns:
            Classification loss
        """
        # Create Alice-signed images (authentic)
        authentic = self.alice([images, message_bits], training=False)

        # Create Eve-forged images (fake authentic)
        forged = self.eve([images, message_bits], training=False)

        # Mixed batch: authentic and forged
        mixed = tf.concat([authentic, forged], axis=0)
        labels = tf.concat(
            [tf.ones((tf.shape(images)[0], 1)), tf.zeros((tf.shape(images)[0], 1))],
            axis=0,
        )

        with tf.GradientTape() as tape:
            _, auth_pred = self.bob(mixed, training=True)
            from .losses import authentication_loss

            loss = authentication_loss(auth_pred, labels)

        gradients = tape.gradient(loss, self.bob.trainable_variables)
        self.optimizer_ab.apply_gradients(zip(gradients, self.bob.trainable_variables))

        return loss

    def get_aes_cipher(self) -> AESCipher:
        """Get the AES cipher used for training."""
        return self.aes_cipher

    def set_learning_rate(self, lr: float) -> None:
        """Update learning rate for both optimizers."""
        self.optimizer_ab.learning_rate.assign(lr)
        self.optimizer_eve.learning_rate.assign(lr)
