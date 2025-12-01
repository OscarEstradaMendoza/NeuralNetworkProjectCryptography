"""
Alice Network - U-Net Encoder-Decoder for watermark embedding.

Alice takes an image and encrypted message bits, and produces
an imperceptible perturbation that embeds the watermark.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from ..config import IMAGE_SIZE, CHANNELS, MESSAGE_LENGTH, PERTURBATION_SCALE, PERTURBATION_BOUND


class PerturbationClipLayer(layers.Layer):
    """
    Custom layer to scale and clip perturbation values.
    
    This replaces Lambda layer to avoid serialization issues with tf in lambda functions.
    """
    def __init__(self, scale, bound, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
        self.bound = bound
    
    def call(self, inputs):
        return tf.clip_by_value(
            inputs * self.scale,
            -self.bound,
            self.bound
        )
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'scale': self.scale,
            'bound': self.bound
        })
        return config


def create_alice_network():
    """
    Create Alice network (U-Net encoder-decoder).
    
    Architecture:
    - Input: Image (64×64×3) + Message (64×64×1) = (64×64×4)
    - Encoder: 4 downsampling blocks
    - Bottleneck: 512 filters at 8×8
    - Decoder: 3 upsampling blocks with skip connections
    - Output: Perturbation map (64×64×3)
    
    Returns:
        Keras Model with two inputs: (image, message_bits)
    """
    # Input layers
    image_input = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS), name='image')
    message_input = keras.Input(shape=(MESSAGE_LENGTH,), name='message_bits')
    
    # Expand message to spatial dimensions (64×64×1)
    message_expanded = layers.Reshape((1, 1, MESSAGE_LENGTH))(message_input)
    message_expanded = layers.UpSampling2D(size=(IMAGE_SIZE, IMAGE_SIZE))(message_expanded)
    
    # Concatenate image and message
    combined = layers.Concatenate(axis=-1)([image_input, message_expanded])  # (64×64×4)
    
    # Encoder path with skip connections
    # Block 1: 64×64 → 64×64
    e1 = layers.Conv2D(64, 3, padding='same')(combined)
    e1 = layers.BatchNormalization()(e1)
    e1 = layers.LeakyReLU(negative_slope=0.2)(e1)
    
    # Block 2: 64×64 → 32×32
    e2 = layers.Conv2D(64, 3, strides=2, padding='same')(e1)
    e2 = layers.BatchNormalization()(e2)
    e2 = layers.LeakyReLU(negative_slope=0.2)(e2)
    e2 = layers.Conv2D(128, 3, padding='same')(e2)
    e2 = layers.BatchNormalization()(e2)
    e2 = layers.LeakyReLU(negative_slope=0.2)(e2)
    
    # Block 3: 32×32 → 16×16
    e3 = layers.Conv2D(128, 3, strides=2, padding='same')(e2)
    e3 = layers.BatchNormalization()(e3)
    e3 = layers.LeakyReLU(negative_slope=0.2)(e3)
    e3 = layers.Conv2D(256, 3, padding='same')(e3)
    e3 = layers.BatchNormalization()(e3)
    e3 = layers.LeakyReLU(negative_slope=0.2)(e3)
    
    # Block 4: 16×16 → 8×8
    e4 = layers.Conv2D(256, 3, strides=2, padding='same')(e3)
    e4 = layers.BatchNormalization()(e4)
    e4 = layers.LeakyReLU(negative_slope=0.2)(e4)
    e4 = layers.Conv2D(256, 3, padding='same')(e4)
    e4 = layers.BatchNormalization()(e4)
    e4 = layers.LeakyReLU(negative_slope=0.2)(e4)
    
    # Bottleneck: 8×8 → 8×8
    bottleneck = layers.Conv2D(512, 3, padding='same')(e4)
    bottleneck = layers.BatchNormalization()(bottleneck)
    bottleneck = layers.LeakyReLU(negative_slope=0.2)(bottleneck)
    
    # Decoder path with skip connections
    # Block 1: 8×8 → 16×16 (connect with e3 which is 16×16)
    d1 = layers.UpSampling2D(size=(2, 2))(bottleneck)  # 8×8 → 16×16
    d1 = layers.Conv2D(256, 3, padding='same')(d1)  # Match e3 channels
    d1 = layers.Concatenate(axis=-1)([d1, e3])  # e3 is 16×16
    d1 = layers.Conv2D(256, 3, padding='same')(d1)
    d1 = layers.BatchNormalization()(d1)
    d1 = layers.LeakyReLU(negative_slope=0.2)(d1)
    
    # Block 2: 16×16 → 32×32 (connect with e2 which is 32×32)
    d2 = layers.UpSampling2D(size=(2, 2))(d1)  # 16×16 → 32×32
    d2 = layers.Conv2D(128, 3, padding='same')(d2)  # Match e2 channels
    d2 = layers.Concatenate(axis=-1)([d2, e2])  # e2 is 32×32
    d2 = layers.Conv2D(128, 3, padding='same')(d2)
    d2 = layers.BatchNormalization()(d2)
    d2 = layers.LeakyReLU(negative_slope=0.2)(d2)
    
    # Block 3: 32×32 → 64×64 (connect with e1 which is 64×64)
    d3 = layers.UpSampling2D(size=(2, 2))(d2)  # 32×32 → 64×64
    d3 = layers.Conv2D(64, 3, padding='same')(d3)  # Match e1 channels
    d3 = layers.Concatenate(axis=-1)([d3, e1])  # e1 is 64×64
    d3 = layers.Conv2D(64, 3, padding='same')(d3)
    d3 = layers.BatchNormalization()(d3)
    d3 = layers.LeakyReLU(negative_slope=0.2)(d3)
    
    # Output: Perturbation map
    perturbation = layers.Conv2D(CHANNELS, 1, padding='same', activation='tanh')(d3)
    
    # Scale and clip perturbation using custom layer (replaces Lambda for proper serialization)
    perturbation = PerturbationClipLayer(
        scale=PERTURBATION_SCALE,
        bound=PERTURBATION_BOUND
    )(perturbation)
    
    # Add perturbation to original image
    output = layers.Add()([image_input, perturbation])
    
    model = keras.Model(inputs=[image_input, message_input], outputs=output, name='alice')
    return model

