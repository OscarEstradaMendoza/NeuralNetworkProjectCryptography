"""
Eve Network - Adversarial U-Net for forgery attempts.

Eve has the same architecture as Alice but with separate weights.
She attempts to forge signatures that fool Bob.
"""

from .alice import create_alice_network


def create_eve_network():
    """
    Create Eve network (same architecture as Alice but separate weights).
    
    Eve uses the same U-Net structure as Alice but with independent parameters.
    This allows her to learn different strategies for creating forgeries.
    
    Returns:
        Keras Model with two inputs: (image, message_bits)
    """
    # Use the same architecture as Alice but with different name
    # This ensures separate weights
    model = create_alice_network()
    model._name = 'eve'  # Rename to 'eve' for clarity
    
    return model

