"""
Key Manager - AES key generation and storage management.

This module handles ONLY key generation, storage, and loading following SRP.
- Generates random AES keys
- Saves/loads keys to/from files
- Manages key storage directory
"""

import os
from Crypto.Random import get_random_bytes
from typing import Optional


class KeyManager:
    """Manage AES keys for the authentication system."""

    def __init__(self, key_dir: str = "keys"):
        """
        Initialize key manager with a storage directory.

        Args:
            key_dir: Directory to store keys. Created if it doesn't exist.
        """
        self.key_dir = key_dir
        os.makedirs(key_dir, exist_ok=True)

    def generate_key(self, key_size: int = 16) -> bytes:
        """
        Generate a random AES key.

        Args:
            key_size: Key size in bytes (16=AES-128, 24=AES-192, 32=AES-256)

        Returns:
            Random key of specified size

        Raises:
            AssertionError: If key_size is not 16, 24, or 32
        """
        assert key_size in [16, 24, 32], (
            f"Key size must be 16, 24, or 32 bytes (got {key_size})"
        )
        return get_random_bytes(key_size)

    def save_key(self, key: bytes, name: str) -> str:
        """
        Save key to file in the key directory.

        Args:
            key: Key bytes to save
            name: Name for the key (file will be saved as name.key)

        Returns:
            Full path to the saved key file

        Raises:
            ValueError: If key is empty
            IOError: If file write fails
        """
        if not key:
            raise ValueError("Key cannot be empty")

        path = os.path.join(self.key_dir, f"{name}.key")
        try:
            with open(path, "wb") as f:
                f.write(key)
        except IOError as e:
            raise IOError(f"Failed to save key to {path}: {e}")

        return path

    def load_key(self, name: str) -> bytes:
        """
        Load key from file in the key directory.

        Args:
            name: Name of the key (file should be name.key)

        Returns:
            Key bytes

        Raises:
            FileNotFoundError: If key file doesn't exist
            IOError: If file read fails
        """
        path = os.path.join(self.key_dir, f"{name}.key")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Key file not found: {path}")

        try:
            with open(path, "rb") as f:
                return f.read()
        except IOError as e:
            raise IOError(f"Failed to load key from {path}: {e}")

    def delete_key(self, name: str) -> bool:
        """
        Delete a key file.

        Args:
            name: Name of the key to delete

        Returns:
            True if deleted, False if file didn't exist

        Raises:
            IOError: If deletion fails
        """
        path = os.path.join(self.key_dir, f"{name}.key")

        if not os.path.exists(path):
            return False

        try:
            os.remove(path)
            return True
        except IOError as e:
            raise IOError(f"Failed to delete key {path}: {e}")

    def list_keys(self) -> list:
        """
        List all stored keys (without .key extension).

        Returns:
            List of key names
        """
        keys = []
        if os.path.exists(self.key_dir):
            for filename in os.listdir(self.key_dir):
                if filename.endswith(".key"):
                    keys.append(filename[:-4])  # Remove .key extension
        return keys

    def key_exists(self, name: str) -> bool:
        """
        Check if a key file exists.

        Args:
            name: Name of the key

        Returns:
            True if key file exists, False otherwise
        """
        path = os.path.join(self.key_dir, f"{name}.key")
        return os.path.exists(path)

    def get_key_dir(self) -> str:
        """Get the key storage directory path."""
        return self.key_dir

    def set_key_dir(self, key_dir: str) -> None:
        """
        Set a new key storage directory.

        Args:
            key_dir: New directory path. Created if it doesn't exist.
        """
        self.key_dir = key_dir
        os.makedirs(key_dir, exist_ok=True)
