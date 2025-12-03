"""
Evaluation Analysis Script

This script performs comprehensive evaluation of the neural image authentication system:
1. Analyzes performance vs training epochs
2. Analyzes performance vs message bit lengths

Generates visualizations showing:
- Authenticity rate
- Average confidence level
- Bit error rate
- Decryption success rate
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import tensorflow as tf

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neural_image_auth.main import train_and_save
from neural_image_auth.inference import NeuralImageAuthenticator
from neural_image_auth.data.datagen import DataPipeline, ImageGenerator
from neural_image_auth.crypto.key_manager import KeyManager
from neural_image_auth.utils import load_model, load_training_config
from neural_image_auth.config import (
    IMAGE_SIZE,
    CHANNELS,
    BATCH_SIZE,
    KEY_DIR,
    LOG_DIR,
    MESSAGE_LENGTH,
    ALICE_BOB_ITERATIONS,
    EVE_ITERATIONS,
)

# Set TensorFlow logging and configure device
from neural_image_auth.device_setup import configure_device
configure_device()


def generate_test_dataset(num_images: int = 50) -> Tuple[np.ndarray, List[str]]:
    """
    Generate a fixed test dataset for consistent evaluation.

    Args:
        num_images: Number of test images to generate

    Returns:
        Tuple of (images, messages) where images is (num_images, 64, 64, 3)
        and messages is a list of strings
    """
    generator = ImageGenerator(IMAGE_SIZE, CHANNELS)
    
    # Generate diverse test images
    images = []
    messages = []
    
    # Mix different image types
    random_imgs = generator.generate_random_images(num_images // 3)
    pattern_imgs = generator.generate_pattern_images(num_images // 3)
    gaussian_imgs = generator.generate_gaussian_images(num_images - 2 * (num_images // 3))
    
    all_images = np.vstack([random_imgs, pattern_imgs, gaussian_imgs])
    
    # Generate unique messages
    for i in range(num_images):
        messages.append(f"TEST_MSG_{i:04d}")
    
    return all_images[:num_images], messages[:num_images]


def evaluate_model(
    authenticator: NeuralImageAuthenticator,
    test_images: np.ndarray,
    test_messages: List[str],
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Evaluate a trained model on test data.

    Args:
        authenticator: Trained NeuralImageAuthenticator instance
        test_images: Test images array (N, 64, 64, 3)
        test_messages: List of test messages (N strings)
        verbose: If True, print progress

    Returns:
        Dictionary with evaluation metrics:
        - authenticity_rate: Fraction of images correctly identified as authentic
        - avg_confidence: Average confidence score
        - avg_bit_error_rate: Average bit error rate
        - decryption_success_rate: Fraction of messages successfully decrypted
    """
    results = {
        "authenticity_rate": [],
        "confidence_scores": [],
        "bit_error_rates": [],
        "decryption_success": [],
    }

    num_images = len(test_images)
    if verbose:
        print(f"Evaluating on {num_images} test images...")

    for i, (image, message) in enumerate(zip(test_images, test_messages)):
        try:
            # Sign image
            signed = authenticator.sign_image(image, message)

            # Verify image
            verification = authenticator.verify_image(signed, threshold=0.5)

            results["authenticity_rate"].append(verification["is_authentic"])
            results["confidence_scores"].append(verification["confidence"])
            results["bit_error_rates"].append(verification["bit_error_rate"])

            # Check if decryption succeeded
            extracted = verification.get("extracted_message")
            decryption_success = extracted is not None and extracted == message
            results["decryption_success"].append(decryption_success)

            if verbose and (i + 1) % 20 == 0:
                print(f"  Processed {i+1}/{num_images} images...")

        except Exception as e:
            if verbose:
                print(f"  Error processing image {i}: {e}")
            # Count as failure
            results["authenticity_rate"].append(False)
            results["confidence_scores"].append(0.0)
            results["bit_error_rates"].append(1.0)
            results["decryption_success"].append(False)

    # Aggregate metrics
    metrics = {
        "authenticity_rate": float(np.mean(results["authenticity_rate"])),
        "avg_confidence": float(np.mean(results["confidence_scores"])),
        "avg_bit_error_rate": float(np.mean(results["bit_error_rates"])),
        "decryption_success_rate": float(np.mean(results["decryption_success"])),
    }

    return metrics


def epoch_analysis(
    epoch_values: List[int],
    num_test_images: int = 50,  # Reduced for M1 Pro memory efficiency
    save_dir: str = "evaluation_results",
    verbose: bool = True,
) -> Dict:
    """
    Train models with different epoch counts and evaluate performance.

    Args:
        epoch_values: List of epoch counts to test (e.g., [1, 2, 5, 10, 20])
        num_test_images: Number of test images to use
        save_dir: Directory to save results
        verbose: If True, print progress

    Returns:
        Dictionary with results for each epoch count
    """
    os.makedirs(save_dir, exist_ok=True)

    # Generate fixed test dataset
    if verbose:
        print("Generating test dataset...")
    test_images, test_messages = generate_test_dataset(num_test_images)

    results = []

    for epochs in epoch_values:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training with {epochs} epochs")
            print(f"{'='*60}\n")

        try:
            # Train model
            alice, bob, eve, aes_key, log_dir = train_and_save(
                num_epochs=epochs,
                num_alice_bob_iters=ALICE_BOB_ITERATIONS,
                num_eve_iters=EVE_ITERATIONS,
                return_models=True,
            )

            # Create authenticator
            authenticator = NeuralImageAuthenticator(alice, bob, aes_key=aes_key)

            # Evaluate
            if verbose:
                print(f"\nEvaluating model trained with {epochs} epochs...")
            metrics = evaluate_model(authenticator, test_images, test_messages, verbose=verbose)

            # Store results
            result = {
                "epochs": epochs,
                "log_dir": log_dir,
                "metrics": metrics,
            }
            results.append(result)

            if verbose:
                print(f"\nResults for {epochs} epochs:")
                print(f"  Authenticity Rate:     {metrics['authenticity_rate']:.3%}")
                print(f"  Avg Confidence:         {metrics['avg_confidence']:.3f}")
                print(f"  Avg Bit Error Rate:     {metrics['avg_bit_error_rate']:.3%}")
                print(f"  Decryption Success:      {metrics['decryption_success_rate']:.3%}")

        except Exception as e:
            if verbose:
                print(f"Error training with {epochs} epochs: {e}")
            continue

    # Save results
    results_file = os.path.join(save_dir, "epoch_analysis_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    if verbose:
        print(f"\nResults saved to {results_file}")

    return {"epoch_analysis": results}


def message_length_analysis(
    message_string_lengths: List[int],
    num_epochs: int = 5,
    num_test_images: int = 50,  # Reduced for M1 Pro memory efficiency
    save_dir: str = "evaluation_results",
    verbose: bool = True,
) -> Dict:
    """
    Test performance with different message string lengths.

    Note: This tests with different message string lengths (in characters),
    which will be padded by AES encryption. The network architecture remains
    fixed at MESSAGE_LENGTH bits. This shows how message size affects
    performance within the fixed architecture.

    Args:
        message_string_lengths: List of message string lengths to test (e.g., [8, 16, 32, 64])
        num_epochs: Number of epochs to train the model (only trains once)
        num_test_images: Number of test images to use
        save_dir: Directory to save results
        verbose: If True, print progress

    Returns:
        Dictionary with results for each message string length
    """
    os.makedirs(save_dir, exist_ok=True)

    # Train a single model first (or load existing)
    if verbose:
        print(f"\n{'='*60}")
        print("Training base model for message length analysis")
        print(f"{'='*60}\n")

    try:
        # Train model once
        alice, bob, eve, aes_key, log_dir = train_and_save(
            num_epochs=num_epochs,
            num_alice_bob_iters=ALICE_BOB_ITERATIONS,
            num_eve_iters=EVE_ITERATIONS,
            return_models=True,
        )

        # Create authenticator
        authenticator = NeuralImageAuthenticator(alice, bob, aes_key=aes_key)

    except Exception as e:
        if verbose:
            print(f"Error training model: {e}")
            import traceback
            traceback.print_exc()
        return {"message_length_analysis": []}

    # Generate test images
    if verbose:
        print("Generating test dataset...")
    test_images, _ = generate_test_dataset(num_test_images)

    results = []

    for msg_str_len in message_string_lengths:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Testing with message string length: {msg_str_len} characters")
            print(f"{'='*60}\n")

        try:
            # Generate test messages of specified length
            test_messages = []
            for i in range(num_test_images):
                # Create message of specified length
                msg = f"MSG_{i:04d}".ljust(msg_str_len, 'X')[:msg_str_len]
                test_messages.append(msg)

            # Evaluate with these message lengths
            if verbose:
                print(f"Evaluating with message length {msg_str_len}...")
            metrics = evaluate_model(authenticator, test_images, test_messages, verbose=verbose)

            # Calculate actual bit length after encryption
            # AES-CBC with IV: IV (16 bytes) + encrypted message (padded to block size)
            from neural_image_auth.crypto.aes_cipher import AESCipher
            cipher = AESCipher(aes_key)
            sample_bits = cipher.encrypt_to_bits(test_messages[0])
            actual_bit_length = len(sample_bits)

            # Store results
            result = {
                "message_string_length": msg_str_len,
                "actual_bit_length": actual_bit_length,
                "log_dir": log_dir,
                "metrics": metrics,
            }
            results.append(result)

            if verbose:
                print(f"\nResults for message string length {msg_str_len} ({actual_bit_length} bits after encryption):")
                print(f"  Authenticity Rate:     {metrics['authenticity_rate']:.3%}")
                print(f"  Avg Confidence:         {metrics['avg_confidence']:.3f}")
                print(f"  Avg Bit Error Rate:     {metrics['avg_bit_error_rate']:.3%}")
                print(f"  Decryption Success:      {metrics['decryption_success_rate']:.3%}")

        except Exception as e:
            if verbose:
                print(f"Error testing with message length {msg_str_len}: {e}")
                import traceback
                traceback.print_exc()
            continue

    # Save results
    results_file = os.path.join(save_dir, "message_length_analysis_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    if verbose:
        print(f"\nResults saved to {results_file}")

    return {"message_length_analysis": results}


def plot_epoch_analysis(results: Dict, save_path: str = "evaluation_results/epoch_analysis.png"):
    """
    Create visualization plots for epoch analysis.

    Args:
        results: Results dictionary from epoch_analysis()
        save_path: Path to save the plot
    """
    epoch_data = results.get("epoch_analysis", [])
    if not epoch_data:
        print("No epoch analysis data to plot")
        return

    epochs = [r["epochs"] for r in epoch_data]
    metrics = {
        "authenticity_rate": [r["metrics"]["authenticity_rate"] for r in epoch_data],
        "avg_confidence": [r["metrics"]["avg_confidence"] for r in epoch_data],
        "avg_bit_error_rate": [r["metrics"]["avg_bit_error_rate"] for r in epoch_data],
        "decryption_success_rate": [r["metrics"]["decryption_success_rate"] for r in epoch_data],
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Performance Metrics vs Training Epochs", fontsize=16, fontweight="bold")

    # Authenticity Rate
    axes[0, 0].plot(epochs, metrics["authenticity_rate"], marker="o", linewidth=2, markersize=8)
    axes[0, 0].set_title("Authenticity Rate", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Epochs")
    axes[0, 0].set_ylabel("Rate")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1.05])

    # Average Confidence
    axes[0, 1].plot(epochs, metrics["avg_confidence"], marker="s", linewidth=2, markersize=8, color="green")
    axes[0, 1].set_title("Average Confidence Level", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Epochs")
    axes[0, 1].set_ylabel("Confidence")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1.05])

    # Bit Error Rate
    axes[1, 0].plot(epochs, metrics["avg_bit_error_rate"], marker="^", linewidth=2, markersize=8, color="red")
    axes[1, 0].set_title("Average Bit Error Rate", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Epochs")
    axes[1, 0].set_ylabel("Error Rate")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, max(0.1, max(metrics["avg_bit_error_rate"]) * 1.1)])

    # Decryption Success Rate
    axes[1, 1].plot(epochs, metrics["decryption_success_rate"], marker="d", linewidth=2, markersize=8, color="purple")
    axes[1, 1].set_title("Decryption Success Rate", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Epochs")
    axes[1, 1].set_ylabel("Success Rate")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1.05])

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {save_path}")
    plt.close()


def plot_message_length_analysis(
    results: Dict, save_path: str = "evaluation_results/message_length_analysis.png"
):
    """
    Create visualization plots for message length analysis.

    Args:
        results: Results dictionary from message_length_analysis()
        save_path: Path to save the plot
    """
    msg_len_data = results.get("message_length_analysis", [])
    if not msg_len_data:
        print("No message length analysis data to plot")
        return

    message_string_lengths = [r["message_string_length"] for r in msg_len_data]
    actual_bit_lengths = [r.get("actual_bit_length", 0) for r in msg_len_data]
    metrics = {
        "authenticity_rate": [r["metrics"]["authenticity_rate"] for r in msg_len_data],
        "avg_confidence": [r["metrics"]["avg_confidence"] for r in msg_len_data],
        "avg_bit_error_rate": [r["metrics"]["avg_bit_error_rate"] for r in msg_len_data],
        "decryption_success_rate": [r["metrics"]["decryption_success_rate"] for r in msg_len_data],
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Performance Metrics vs Message String Length", fontsize=16, fontweight="bold")

    # Authenticity Rate
    axes[0, 0].plot(message_string_lengths, metrics["authenticity_rate"], marker="o", linewidth=2, markersize=8)
    axes[0, 0].set_title("Authenticity Rate", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Message String Length (characters)")
    axes[0, 0].set_ylabel("Rate")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1.05])

    # Average Confidence
    axes[0, 1].plot(message_string_lengths, metrics["avg_confidence"], marker="s", linewidth=2, markersize=8, color="green")
    axes[0, 1].set_title("Average Confidence Level", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Message String Length (characters)")
    axes[0, 1].set_ylabel("Confidence")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1.05])

    # Bit Error Rate
    axes[1, 0].plot(message_string_lengths, metrics["avg_bit_error_rate"], marker="^", linewidth=2, markersize=8, color="red")
    axes[1, 0].set_title("Average Bit Error Rate", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Message String Length (characters)")
    axes[1, 0].set_ylabel("Error Rate")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, max(0.1, max(metrics["avg_bit_error_rate"]) * 1.1)])

    # Decryption Success Rate
    axes[1, 1].plot(message_string_lengths, metrics["decryption_success_rate"], marker="d", linewidth=2, markersize=8, color="purple")
    axes[1, 1].set_title("Decryption Success Rate", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Message String Length (characters)")
    axes[1, 1].set_ylabel("Success Rate")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1.05])

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {save_path}")
    plt.close()


def main():
    """Main function to run evaluation analysis."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate neural image authentication system")
    parser.add_argument(
        "--epoch-analysis",
        action="store_true",
        help="Run epoch analysis (train with different epoch counts)",
    )
    parser.add_argument(
        "--message-length-analysis",
        action="store_true",
        help="Run message length analysis (train with different message bit lengths)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=[1, 2, 5, 10],
        help="Epoch values to test (default: [1, 2, 5, 10])",
    )
    parser.add_argument(
        "--message-lengths",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64],
        help="Message string lengths (characters) to test (default: [8, 16, 32, 64])",
    )
    parser.add_argument(
        "--num-test-images",
        type=int,
        default=50,
        help="Number of test images to use (default: 50, optimized for M1 Pro)",
    )
    parser.add_argument(
        "--num-epochs-for-message-length",
        type=int,
        default=5,
        help="Number of epochs to train for message length analysis (default: 5)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="evaluation_results",
        help="Directory to save results (default: evaluation_results)",
    )

    args = parser.parse_args()

    # If no analysis specified, run both
    if not args.epoch_analysis and not args.message_length_analysis:
        args.epoch_analysis = True
        args.message_length_analysis = True

    all_results = {}

    # Run epoch analysis
    if args.epoch_analysis:
        print("\n" + "=" * 60)
        print("EPOCH ANALYSIS")
        print("=" * 60)
        epoch_results = epoch_analysis(
            epoch_values=args.epochs,
            num_test_images=args.num_test_images,
            save_dir=args.save_dir,
            verbose=True,
        )
        all_results.update(epoch_results)

        # Plot results
        plot_epoch_analysis(epoch_results, os.path.join(args.save_dir, "epoch_analysis.png"))

    # Run message length analysis
    if args.message_length_analysis:
        print("\n" + "=" * 60)
        print("MESSAGE LENGTH ANALYSIS")
        print("=" * 60)
        msg_len_results = message_length_analysis(
            message_string_lengths=args.message_lengths,
            num_epochs=args.num_epochs_for_message_length,
            num_test_images=args.num_test_images,
            save_dir=args.save_dir,
            verbose=True,
        )
        all_results.update(msg_len_results)

        # Plot results
        plot_message_length_analysis(
            msg_len_results, os.path.join(args.save_dir, "message_length_analysis.png")
        )

    # Save combined results
    combined_file = os.path.join(args.save_dir, "all_evaluation_results.json")
    with open(combined_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to {combined_file}")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

