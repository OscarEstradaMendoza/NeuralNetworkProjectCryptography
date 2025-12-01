"""
Inference API for signing and verifying images with AES-encrypted watermarks.

This module handles ONLY inference/prediction operations following SRP.
Provides high-level API for:
- Signing images with encrypted watermarks
- Verifying image authenticity and extracting messages
"""

import numpy as np
import tensorflow as tf
from typing import Dict, Optional
from .crypto.aes_cipher import AESCipher
from .data.preprocessing import preprocess_for_network, postprocess_from_network
from .config import MESSAGE_LENGTH


class NeuralImageAuthenticator:
    """
    Main API for signing and verifying images with AES-encrypted watermarks.

    This class combines Alice (signer), Bob (verifier), and AES encryption
    into a unified interface for image authentication.
    """

    def __init__(
        self,
        alice_model,
        bob_model,
        aes_key: Optional[bytes] = None,
    ):
        """
        Initialize the authenticator with pre-trained models.

        Args:
            alice_model: Trained Alice (encoder) network
            bob_model: Trained Bob (decoder/classifier) network
            aes_key: AES key for encryption. If None, generates random key.
        """
        self.alice = alice_model
        self.bob = bob_model
        self.cipher = AESCipher(aes_key)

    def sign_image(
        self,
        image: np.ndarray,
        message: str = "AUTHENTIC",
    ) -> np.ndarray:
        """
        Embed encrypted message into image as imperceptible watermark.

        The watermark is created through:
        1. AES-CBC encryption of the message
        2. Conversion to binary array
        3. Embedding by Alice network

        Args:
            image: Original image (any shape, will be resized to 64×64)
                   Values should be in [0, 1] or [0, 255] or [-1, 1]
            message: Message to embed (default "AUTHENTIC")
                     Will be AES-encrypted before embedding

        Returns:
            Signed image with imperceptible watermark (64×64×3)
            Values in [0, 255] as uint8 for saving

        Raises:
            ValueError: If image format is invalid
            RuntimeError: If Alice model is not available
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be numpy array")

        # Preprocess image
        image_tensor = preprocess_for_network(image)

        # Add batch dimension
        image_tensor = np.expand_dims(image_tensor, axis=0)
        image_tensor = tf.constant(image_tensor)

        # Encrypt message to bits
        message_bits = self.cipher.encrypt_to_bits(message)

        # Add batch dimension to message bits
        message_bits = np.expand_dims(message_bits, axis=0)
        message_bits = tf.constant(message_bits)

        # Embed with Alice
        signed = self.alice([image_tensor, message_bits], training=False)

        # Remove batch dimension and convert to output format
        signed = signed[0].numpy()
        signed = postprocess_from_network(signed)

        return signed

    def verify_image(
        self,
        image: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict:
        """
        Verify image authenticity and attempt to extract embedded message.

        Returns a dictionary with:
        - Authenticity verdict (True/False)
        - Confidence score (0-1)
        - Extracted message (if decryption succeeds)
        - Bit error rate (if Bob extracts bits)

        Args:
            image: Image to verify (any shape, will be resized to 64×64)
                   Values should be in [0, 1] or [0, 255] or [-1, 1]
            threshold: Classification threshold for authenticity (default 0.5)

        Returns:
            Dictionary with keys:
                - 'is_authentic': bool, whether image passes authentication
                - 'confidence': float in [0, 1], Bob's confidence score
                - 'extracted_message': str or None, decrypted message if available
                - 'bit_error_rate': float, estimate of bit corruption
                - 'extracted_bits': ndarray, raw extracted bits (for debugging)

        Raises:
            ValueError: If image format is invalid
            RuntimeError: If Bob model is not available
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be numpy array")

        # Preprocess image
        image_tensor = preprocess_for_network(image)

        # Add batch dimension
        image_tensor = np.expand_dims(image_tensor, axis=0)
        image_tensor = tf.constant(image_tensor)

        # Bob extracts and classifies
        extracted_bits, auth_prob = self.bob(image_tensor, training=False)

        # Remove batch dimension
        extracted_bits = extracted_bits[0].numpy()
        auth_prob = float(auth_prob[0, 0])

        is_authentic = auth_prob > threshold

        # Try to decrypt message
        extracted_message = None
        try:
            extracted_message = self.cipher.decrypt_from_bits(extracted_bits)
        except Exception as e:
            # Decryption failed - likely due to tampering or wrong key
            pass

        # Estimate bit error rate
        # BER = (1 - accuracy of bit thresholding)
        hard_bits = np.sign(extracted_bits)
        bit_error_rate = float(np.mean(hard_bits != extracted_bits))

        return {
            "is_authentic": bool(is_authentic),
            "confidence": auth_prob,
            "extracted_message": extracted_message,
            "bit_error_rate": bit_error_rate,
            "extracted_bits": extracted_bits,  # Raw bits for debugging
        }

    def get_aes_key(self) -> bytes:
        """Get the AES key used for encryption."""
        return self.cipher.get_key()

    def set_aes_key(self, key: bytes) -> None:
        """
        Set a new AES key.

        Useful for changing the key without creating a new authenticator.

        Args:
            key: 16, 24, or 32 byte AES key

        Raises:
            ValueError: If key length is invalid
        """
        self.cipher.set_key(key)

    def batch_sign_images(
        self,
        images: np.ndarray,
        message: str = "AUTHENTIC",
    ) -> np.ndarray:
        """
        Sign a batch of images efficiently.

        Args:
            images: Array of images of shape (batch_size, height, width, 3)
            message: Message to embed (same for all images)

        Returns:
            Array of signed images of shape (batch_size, 64, 64, 3)
        """
        signed_images = []

        for image in images:
            signed = self.sign_image(image, message)
            signed_images.append(signed)

        return np.array(signed_images)

    def batch_verify_images(
        self,
        images: np.ndarray,
        threshold: float = 0.5,
    ) -> list:
        """
        Verify a batch of images efficiently.

        Args:
            images: Array of images of shape (batch_size, height, width, 3)
            threshold: Classification threshold for authenticity

        Returns:
            List of result dictionaries (one per image)
        """
        results = []

        for image in images:
            result = self.verify_image(image, threshold)
            results.append(result)

        return results
