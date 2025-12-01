"""
Models package for neural image authentication system.

Contains:
- Alice: U-Net encoder-decoder for watermark embedding
- Bob: CNN with dual heads for message extraction and authentication
- Eve: Adversarial U-Net for forgery attempts
"""

from .alice import create_alice_network
from .bob import create_bob_network
from .eve import create_eve_network

__all__ = ['create_alice_network', 'create_bob_network', 'create_eve_network']

