"""
Configuration module - Hyperparameters and constants for the neural image authentication system.

This module follows SRP (Single Responsibility Principle) by ONLY containing configuration values.
All constants are defined here for easy modification and centralized management.
"""

import os

# ============================================================================
# IMAGE SETTINGS
# ============================================================================
IMAGE_SIZE = 64
CHANNELS = 3

# ============================================================================
# MESSAGE/WATERMARK SETTINGS
# ============================================================================
# Total bits after AES encryption with IV (16 bytes IV + encrypted message)
MESSAGE_LENGTH = 256

# Raw message length before encryption (16 bytes = 128 bits)
# This allows encrypting 16-byte messages (e.g., 16-character strings)
RAW_MESSAGE_LENGTH = 128

# ============================================================================
# AES SETTINGS
# ============================================================================
# Key size: 16 (AES-128), 24 (AES-192), or 32 (AES-256) bytes
AES_KEY_SIZE = 16

# Encryption mode: CBC is more secure than ECB for images
AES_MODE = "CBC"

# ============================================================================
# TRAINING SETTINGS
# ============================================================================
# Optimized for Apple M1 Pro (16GB RAM)
# Reduced batch size for better memory efficiency
BATCH_SIZE = 16  # Reduced from 32 for M1 Pro memory constraints
LEARNING_RATE = 0.0002
LEARNING_RATE_DECAY = 0.99  # Learning rate decay per epoch

# Outer adversarial loop iterations
ADV_ITERATIONS = 2

# Inner training iterations per phase
# Slightly reduced for faster iteration on M1 Pro
ALICE_BOB_ITERATIONS = 20

# Eve gets 2x iterations to simulate a strong adversary
EVE_ITERATIONS = 40

# Number of validation batches per epoch
VALIDATION_BATCHES = 5

# ============================================================================
# ALICE (ENCODER) SETTINGS
# ============================================================================
# How much perturbation to add (α)
PERTURBATION_SCALE = 0.05

# Maximum L∞ norm of perturbation (ε)
PERTURBATION_BOUND = 0.1

# ============================================================================
# LOSS WEIGHTS (LAMBDA VALUES)
# ============================================================================
# Weight for reconstruction loss (keep perturbed image close to original)
LAMBDA_RECONSTRUCTION = 1.0

# Weight for message extraction loss (Bob should extract correct message)
LAMBDA_MESSAGE = 2.0

# Weight for authentication loss (Bob should classify correctly)
LAMBDA_AUTHENTICATION = 1.0

# Weight for imperceptibility loss (bound perturbation magnitude)
LAMBDA_IMPERCEPTIBILITY = 0.5

# Create a dictionary of lambdas for easier passing to loss functions
LAMBDAS = {
    'reconstruction': LAMBDA_RECONSTRUCTION,
    'message': LAMBDA_MESSAGE,
    'authentication': LAMBDA_AUTHENTICATION,
    'imperceptibility': LAMBDA_IMPERCEPTIBILITY,
}

# ============================================================================
# BOB (DECODER/CLASSIFIER) SETTINGS
# ============================================================================
# Dropout rate in authenticity classification head
BOB_DROPOUT_RATE_1 = 0.5
BOB_DROPOUT_RATE_2 = 0.3

# ============================================================================
# DIRECTORY SETTINGS
# ============================================================================
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(PROJECT_DIR, "models")
LOG_DIR = os.path.join(PROJECT_DIR, "logs")
KEY_DIR = os.path.join(PROJECT_DIR, "keys")
DATA_DIR = os.path.join(PROJECT_DIR, "data")

# Create directories if they don't exist
for directory in [MODEL_DIR, LOG_DIR, KEY_DIR, DATA_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============================================================================
# DEVICE SETTINGS (Optimized for Apple M1 Pro)
# ============================================================================
# Use GPU if available (TensorFlow will auto-detect MPS on Apple Silicon)
USE_GPU = True

# Mixed precision training
# Note: MPS backend has limited mixed precision support, may cause issues
# Set to False if you encounter errors with MPS backend
USE_MIXED_PRECISION = False  # Disabled for M1 Pro MPS compatibility

# Apple Silicon specific settings
# MPS (Metal Performance Shaders) backend for Apple Silicon GPUs
USE_MPS = True  # Use Metal backend on Apple Silicon

# Memory growth settings (prevents TensorFlow from allocating all GPU memory)
MEMORY_GROWTH = True  # Allow memory growth for better memory management

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================
# Number of images to visualize during training
NUM_VIS_IMAGES = 4

# DPI for saved figures
FIGURE_DPI = 100

# ============================================================================
# RANDOM SEED
# ============================================================================
# Set for reproducibility
RANDOM_SEED = 42
