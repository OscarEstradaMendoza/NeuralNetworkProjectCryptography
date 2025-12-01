"""
Main entry point for the neural authentication system.

This script orchestrates the complete workflow:
1. Initialize models (Alice, Bob, Eve)
2. Load/generate AES key
3. Run adversarial training
4. Evaluate performance
5. Save trained models
"""

import os
import sys
import numpy as np
import tensorflow as tf
from typing import Optional, Dict

# Set up TensorFlow
tf.get_logger().setLevel("ERROR")

from .config import (
    IMAGE_SIZE,
    CHANNELS,
    MESSAGE_LENGTH,
    BATCH_SIZE,
    LEARNING_RATE,
    ADV_ITERATIONS,
    ALICE_BOB_ITERATIONS,
    EVE_ITERATIONS,
    RANDOM_SEED,
    MODEL_DIR,
    LOG_DIR,
    KEY_DIR,
    LAMBDAS,
)
from .models.alice import create_alice_network
from .models.bob import create_bob_network
from .models.eve import create_eve_network
from .training.trainer import AdversarialTrainer
from .data.datagen import DataPipeline
from .crypto.key_manager import KeyManager
from .crypto.aes_cipher import AESCipher
from .utils import (
    save_model,
    save_training_config,
    print_model_summary,
    create_log_directory,
    plot_training_history,
)


def set_random_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def initialize_models() -> tuple:
    """
    Initialize Alice, Bob, and Eve networks.

    Returns:
        Tuple of (alice, bob, eve) models
    """
    print("\n" + "=" * 60)
    print("INITIALIZING NEURAL NETWORKS")
    print("=" * 60)

    alice = create_alice_network()
    bob = create_bob_network()
    eve = create_eve_network()

    print_model_summary(alice, "Alice (Encoder)")
    print_model_summary(bob, "Bob (Decoder/Classifier)")
    print_model_summary(eve, "Eve (Adversary)")

    return alice, bob, eve


def initialize_aes_key(key_manager: KeyManager) -> bytes:
    """
    Initialize AES key (generate new or load existing).

    Args:
        key_manager: KeyManager instance

    Returns:
        AES key as bytes
    """
    print("\n" + "=" * 60)
    print("INITIALIZING AES KEY")
    print("=" * 60)

    key_name = "default_key"

    if key_manager.key_exists(key_name):
        print(f"Loading existing key: {key_name}")
        aes_key = key_manager.load_key(key_name)
    else:
        print(f"Generating new key: {key_name}")
        aes_key = key_manager.generate_key(key_size=16)  # AES-128
        key_manager.save_key(aes_key, key_name)

    print(f"AES Key (hex): {aes_key.hex()[:32]}... (truncated)")
    return aes_key


def train_adversarial(
    trainer: AdversarialTrainer,
    data_pipeline: DataPipeline,
    num_epochs: int = ADV_ITERATIONS,
    num_alice_bob_iters: int = ALICE_BOB_ITERATIONS,
    num_eve_iters: int = EVE_ITERATIONS,
    log_dir: Optional[str] = None,
) -> Dict:
    """
    Run adversarial training loop.

    Args:
        trainer: AdversarialTrainer instance
        data_pipeline: DataPipeline instance
        num_epochs: Number of adversarial iterations
        num_alice_bob_iters: Iterations per epoch for Alice+Bob
        num_eve_iters: Iterations per epoch for Eve
        log_dir: Optional directory for logs

    Returns:
        Training history dictionary
    """
    print("\n" + "=" * 60)
    print("STARTING ADVERSARIAL TRAINING")
    print("=" * 60)
    print(f"Total Epochs: {num_epochs}")
    print(f"Alice+Bob Iters/Epoch: {num_alice_bob_iters}")
    print(f"Eve Iters/Epoch: {num_eve_iters}")
    print(f"Batch Size: {BATCH_SIZE}")
    print("=" * 60 + "\n")

    history = {
        "epoch": [],
        "alice_bob_loss": [],
        "bit_accuracy": [],
        "bob_classifier_loss": [],
        "eve_loss": [],
    }

    for epoch in range(num_epochs):
        epoch_losses = {
            "alice_bob": [],
            "bit_accuracy": [],
            "bob_classifier": [],
            "eve": [],
        }

        # Phase 1: Train Alice + Bob
        print(f"Epoch {epoch+1}/{num_epochs} - Phase 1: Alice + Bob Training")
        for step in range(num_alice_bob_iters):
            # Get training batch
            images = data_pipeline.get_training_batch()

            # Generate message bits
            message = f"AUTH_{epoch}_{step}"  # Unique per iteration
            message_bits = trainer.get_aes_cipher().encrypt_to_bits(message)
            message_bits = tf.constant(
                np.tile(message_bits[np.newaxis, :], [BATCH_SIZE, 1])
            )

            # Train step
            loss, loss_dict, acc = trainer.train_step_alice_bob(
                images, message_bits
            )

            epoch_losses["alice_bob"].append(float(loss))
            epoch_losses["bit_accuracy"].append(float(acc))

            if (step + 1) % 5 == 0:
                print(
                    f"  Step {step+1}/{num_alice_bob_iters} | "
                    f"Loss: {loss:.4f} | BER: {1-acc:.3f}"
                )

        # Phase 2: Train Bob Classifier
        print(f"Epoch {epoch+1}/{num_epochs} - Phase 2: Bob Classifier Training")
        for step in range(num_alice_bob_iters):
            images = data_pipeline.get_training_batch()

            message_bits = trainer.get_aes_cipher().encrypt_to_bits(message)
            message_bits = tf.constant(
                np.tile(message_bits[np.newaxis, :], [BATCH_SIZE, 1])
            )

            loss = trainer.train_step_bob_classifier(images, message_bits)
            epoch_losses["bob_classifier"].append(float(loss))

            if (step + 1) % 5 == 0:
                print(
                    f"  Step {step+1}/{num_alice_bob_iters} | "
                    f"Classification Loss: {loss:.4f}"
                )

        # Phase 3: Train Eve
        print(f"Epoch {epoch+1}/{num_epochs} - Phase 3: Eve Training")
        for step in range(num_eve_iters):
            images = data_pipeline.get_training_batch()

            message_bits = trainer.get_aes_cipher().encrypt_to_bits(message)
            message_bits = tf.constant(
                np.tile(message_bits[np.newaxis, :], [BATCH_SIZE, 1])
            )

            loss = trainer.train_step_eve(images, message_bits)
            epoch_losses["eve"].append(float(loss))

            if (step + 1) % 5 == 0:
                print(
                    f"  Step {step+1}/{num_eve_iters} | Eve Loss: {loss:.4f}"
                )

        # Phase 4: Harden Bob
        print(f"Epoch {epoch+1}/{num_epochs} - Phase 4: Hardening Bob against Eve")
        for step in range(num_alice_bob_iters // 2):
            images = data_pipeline.get_training_batch()

            message_bits = trainer.get_aes_cipher().encrypt_to_bits(message)
            message_bits = tf.constant(
                np.tile(message_bits[np.newaxis, :], [BATCH_SIZE, 1])
            )

            loss = trainer.train_step_harden_bob(images, message_bits)

            if (step + 1) % 5 == 0:
                print(f"  Step {step+1} | Hardening Loss: {loss:.4f}")

        # Log epoch statistics
        avg_alice_bob_loss = np.mean(epoch_losses["alice_bob"])
        avg_bit_accuracy = np.mean(epoch_losses["bit_accuracy"])
        avg_bob_classifier_loss = np.mean(epoch_losses["bob_classifier"])
        avg_eve_loss = np.mean(epoch_losses["eve"])

        history["epoch"].append(epoch + 1)
        history["alice_bob_loss"].append(avg_alice_bob_loss)
        history["bit_accuracy"].append(avg_bit_accuracy)
        history["bob_classifier_loss"].append(avg_bob_classifier_loss)
        history["eve_loss"].append(avg_eve_loss)

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Alice+Bob Loss: {avg_alice_bob_loss:.4f}")
        print(f"  Bit Accuracy: {avg_bit_accuracy:.3%}")
        print(f"  Bob Classifier Loss: {avg_bob_classifier_loss:.4f}")
        print(f"  Eve Loss: {avg_eve_loss:.4f}")
        print()

    return history


def save_results(
    alice, bob, eve, history: Dict, aes_key: bytes, log_dir: str
) -> None:
    """
    Save trained models and training history.

    Args:
        alice: Trained Alice model
        bob: Trained Bob model
        eve: Trained Eve model
        history: Training history dictionary
        aes_key: AES key used
        log_dir: Directory to save to
    """
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    # Save models
    save_model(alice, "alice", os.path.join(log_dir, "models"))
    save_model(bob, "bob", os.path.join(log_dir, "models"))
    save_model(eve, "eve", os.path.join(log_dir, "models"))

    # Save training history
    from .utils import save_training_history

    save_training_history(
        history, os.path.join(log_dir, "training_history.json")
    )

    # Save configuration
    config = {
        "image_size": IMAGE_SIZE,
        "channels": CHANNELS,
        "message_length": MESSAGE_LENGTH,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "lambdas": LAMBDAS,
        "aes_key_hex": aes_key.hex(),  # For reference (keep secure!)
    }
    save_training_config(
        config, os.path.join(log_dir, "config.json")
    )

    print(f"\nAll results saved to: {log_dir}")


def main(
    num_epochs: int = ADV_ITERATIONS,
    num_alice_bob_iters: int = ALICE_BOB_ITERATIONS,
    num_eve_iters: int = EVE_ITERATIONS,
) -> None:
    """
    Main training workflow.

    Args:
        num_epochs: Number of adversarial training epochs
        num_alice_bob_iters: Alice+Bob training iterations per epoch
        num_eve_iters: Eve training iterations per epoch
    """
    # Set random seed for reproducibility
    set_random_seed()

    # Create log directory
    log_dir = create_log_directory(LOG_DIR)

    # Initialize models
    alice, bob, eve = initialize_models()

    # Initialize AES key
    key_manager = KeyManager(KEY_DIR)
    aes_key = initialize_aes_key(key_manager)

    # Create trainer
    trainer = AdversarialTrainer(alice, bob, eve, aes_key=aes_key)

    # Create data pipeline
    data_pipeline = DataPipeline(
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        channels=CHANNELS,
    )

    # Run training
    history = train_adversarial(
        trainer,
        data_pipeline,
        num_epochs=num_epochs,
        num_alice_bob_iters=num_alice_bob_iters,
        num_eve_iters=num_eve_iters,
        log_dir=log_dir,
    )

    # Save results
    save_results(alice, bob, eve, history, aes_key, log_dir)

    # Plot training history
    plot_training_history(
        history,
        metrics=["alice_bob_loss", "bit_accuracy", "bob_classifier_loss", "eve_loss"],
        save_path=os.path.join(log_dir, "training_history.png"),
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Results saved to: {log_dir}")


if __name__ == "__main__":
    # Run with default parameters
    # Can be customized via command-line arguments
    main()
