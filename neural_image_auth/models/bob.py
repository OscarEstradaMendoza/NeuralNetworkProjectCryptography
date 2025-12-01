"""
Bob Network - CNN with dual heads for message extraction and authentication.

Bob takes a signed image and:
1. Extracts the embedded message bits
2. Classifies whether the image is authentic
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from ..config import IMAGE_SIZE, CHANNELS, MESSAGE_LENGTH, BOB_DROPOUT_RATE_1, BOB_DROPOUT_RATE_2


def create_bob_network():
    """
    Create Bob network (CNN with dual heads).
    
    Architecture:
    - Input: Image (64×64×3)
    - Shared Encoder: 4 downsampling blocks → 4×4×256 features
    - Head 1 (Message): Dense layers → Tanh output (message_length,)
    - Head 2 (Authentication): Dense layers with dropout → Sigmoid output
    
    Returns:
        Keras Model with dual outputs: (extracted_bits, authenticity_prob)
    """
    # Input layer
    image_input = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS), name='image')
    
    # Shared feature extraction (encoder)
    # Block 1: 64×64 → 32×32
    x = layers.Conv2D(32, 3, padding='same')(image_input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(32, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Block 2: 32×32 → 16×16
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Block 3: 16×16 → 8×8
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(128, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Block 4: 8×8 → 4×4
    x = layers.Conv2D(256, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(256, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Flatten for dense layers
    x = layers.Flatten()(x)
    shared_features = layers.Dense(1024, activation='relu')(x)
    
    # Head 1: Message Extraction
    msg_head = layers.Dense(512, activation='relu')(shared_features)
    msg_head = layers.Dense(256, activation='relu')(msg_head)
    extracted_bits = layers.Dense(MESSAGE_LENGTH, activation='tanh', name='extracted_bits')(msg_head)
    
    # Head 2: Authentication Classification
    auth_head = layers.Dense(256, activation='relu')(shared_features)
    auth_head = layers.Dropout(BOB_DROPOUT_RATE_1)(auth_head)
    auth_head = layers.Dense(128, activation='relu')(auth_head)
    auth_head = layers.Dropout(BOB_DROPOUT_RATE_2)(auth_head)
    authenticity_prob = layers.Dense(1, activation='sigmoid', name='authenticity_prob')(auth_head)
    
    model = keras.Model(
        inputs=image_input,
        outputs=[extracted_bits, authenticity_prob],
        name='bob'
    )
    return model

