"""
AES Cipher - AES-CBC encryption/decryption for watermark data.

This module handles ONLY AES encryption/decryption logic following SRP.
- Encrypts/decrypts plaintext using AES-CBC mode
- Converts encrypted data to/from binary arrays for neural network processing
- Uses random IVs for each encryption (security best practice)
"""

from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
import numpy as np
from typing import Tuple


class AESCipher:
    """AES-CBC encryption/decryption for watermark data."""

    def __init__(self, key: bytes = None):
        """
        Initialize AES cipher with a given or random key.

        Args:
            key: 16, 24, or 32 byte key for AES-128, AES-192, or AES-256.
                 If None, generates a random 16-byte (AES-128) key.

        Raises:
            ValueError: If key length is not 16, 24, or 32 bytes.
        """
        if key is None:
            self.key = get_random_bytes(16)  # Default: AES-128
        else:
            if len(key) not in [16, 24, 32]:
                raise ValueError("Key size must be 16, 24, or 32 bytes")
            self.key = key

        self.block_size = AES.block_size  # 16 bytes for AES

    def encrypt(self, plaintext: bytes) -> Tuple[bytes, bytes]:
        """
        Encrypt plaintext using AES-CBC mode.

        Args:
            plaintext: Data to encrypt (bytes)

        Returns:
            Tuple of (ciphertext, iv):
                - ciphertext: Encrypted data
                - iv: Initialization vector (randomly generated for security)
        """
        iv = get_random_bytes(self.block_size)
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        padded = pad(plaintext, self.block_size)
        ciphertext = cipher.encrypt(padded)
        return ciphertext, iv

    def decrypt(self, ciphertext: bytes, iv: bytes) -> bytes:
        """
        Decrypt ciphertext using AES-CBC mode.

        Args:
            ciphertext: Encrypted data
            iv: Initialization vector used during encryption

        Returns:
            Decrypted plaintext (bytes)

        Raises:
            ValueError: If decryption or unpadding fails (data may be corrupted)
        """
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        try:
            padded = cipher.decrypt(ciphertext)
            plaintext = unpad(padded, self.block_size)
            return plaintext
        except ValueError as e:
            raise ValueError(f"Decryption failed - data may be corrupted: {e}")

    def encrypt_to_bits(self, message: str) -> np.ndarray:
        """
        Encrypt message and convert to binary array for neural network input.

        The binary representation is converted from {0, 1} to {-1, 1} for better
        neural network training (zero-centered).

        Args:
            message: String message to encrypt

        Returns:
            Binary numpy array of shape (total_bits,) with values in {-1, 1}
            where total_bits = 8 * (16 + len(encrypted_data))
            (IV is prepended to ciphertext for self-contained decryption)
        """
        # Encrypt message
        ciphertext, iv = self.encrypt(message.encode("utf-8"))
        # Prepend IV to ciphertext for decryption without external storage
        combined = iv + ciphertext

        # Convert bytes to bits
        bits = []
        for byte in combined:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)

        # Convert {0, 1} to {-1, 1} for neural network
        # This centers the bits around 0, improving gradient flow
        bits_array = np.array(bits, dtype=np.float32)
        bits_array = bits_array * 2 - 1  # Map [0, 1] to [-1, 1]

        return bits_array

    def decrypt_from_bits(self, bits: np.ndarray) -> str:
        """
        Convert binary array back to decrypted message.

        Reverses the encrypt_to_bits process:
        1. Convert bits from {-1, 1} back to {0, 1}
        2. Convert bits to bytes
        3. Split IV and ciphertext
        4. Decrypt using AES-CBC

        Args:
            bits: Binary array with values in {-1, 1} (output from Bob decoder)

        Returns:
            Decrypted string message

        Raises:
            ValueError: If decryption fails (wrong key, corrupted bits, or tampering)
        """
        # Convert {-1, 1} back to {0, 1}
        bits = np.clip(bits, -1, 1)  # Clip to valid range in case of noise
        bits = ((bits + 1) / 2).astype(np.uint8)

        # Convert bits to bytes
        byte_array = bytearray()
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                if i + j < len(bits):
                    byte = (byte << 1) | bits[i + j]
            byte_array.append(byte)

        # Split IV and ciphertext
        iv = bytes(byte_array[: self.block_size])
        ciphertext = bytes(byte_array[self.block_size :])

        # Decrypt
        plaintext = self.decrypt(ciphertext, iv)
        return plaintext.decode("utf-8")

    def get_key(self) -> bytes:
        """Get the current AES key."""
        return self.key

    def set_key(self, key: bytes) -> None:
        """
        Set a new AES key.

        Args:
            key: 16, 24, or 32 byte key

        Raises:
            ValueError: If key length is invalid
        """
        if len(key) not in [16, 24, 32]:
            raise ValueError("Key size must be 16, 24, or 32 bytes")
        self.key = key
