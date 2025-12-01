# Neural Image Authentication System with AES Encryption

A comprehensive deep learning-based image authentication system that combines adversarial neural networks with AES-CBC encryption for the CS4379H Cryptography course project.

## System Overview

This system implements a hybrid cryptographic authentication scheme where:

1. **Alice (Encoder)**: Embeds AES-encrypted watermarks into images as imperceptible perturbations
2. **Bob (Decoder/Classifier)**: Extracts and verifies watermarks, classifying images as authentic or forged
3. **Eve (Adversary)**: Attempts to forge signatures or extract messages without the AES key

The system is trained adversarially where Alice and Bob cooperate against Eve, who tries to break the scheme.

## Key Features

**AES-CBC Encryption**: Secure message encryption using industry-standard cryptography  
**Neural Watermarking**: Imperceptible perturbations using U-Net architectures  
**Dual-headed Bob**: Simultaneous message extraction and authenticity classification  
**Adversarial Training**: Robust against forgery attempts by trained Eve network  
**SOLID Design**: Modular architecture with clear separation of concerns  
**Comprehensive Metrics**: BER, PSNR, authentication accuracy tracking

## System Architecture

```
SIGNING FLOW:
                                    ┌─────────────┐
   Secret Message ──► AES Encrypt ──►│             │
                         │          │    Alice    │──► Signed Image
   Original Image ───────┼─────────►│  (Encoder)  │
                         │          └─────────────┘
                    AES Key (shared secret)

VERIFICATION FLOW:
                      ┌─────────────┐
   Received Image ───►│     Bob     │──► Extracted Bits ──► AES Decrypt ──► Message
                      │  (Decoder)  │                            │
                      └─────────────┘                       AES Key
                            │
                            ▼
                    Authenticity Score (0-1)
```

## Project Structure

```
neural_image_auth/
├── config.py              # Hyperparameters and constants
├── crypto/
│   ├── __init__.py
│   ├── aes_cipher.py      # AES-CBC encryption/decryption
│   └── key_manager.py     # Key generation and storage
├── data/
│   ├── __init__.py
│   ├── datagen.py         # Synthetic image generation
│   └── preprocessing.py   # Image preprocessing utilities
├── models/
│   ├── __init__.py
│   ├── alice.py           # Alice encoder network (U-Net)
│   ├── bob.py             # Bob decoder/classifier network (CNN)
│   └── eve.py             # Eve adversary network (U-Net)
├── training/
│   ├── __init__.py
│   ├── losses.py          # Loss function definitions
│   ├── trainer.py         # Training loop orchestration
│   └── metrics.py         # Evaluation metrics
├── inference.py           # Inference/prediction API
├── utils.py               # Helper functions
├── main.py                # Entry point
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

```bash
# Clone repository
git clone https://github.com/ChrissMollina/CS4379H_ProjectV2_Adversarial-Neural-Cryptography.git
cd CS4379H_ProjectV2_Adversarial-Neural-Cryptography

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Training the System

```python
from neural_image_auth.main import main
from neural_image_auth.config import ADV_ITERATIONS, ALICE_BOB_ITERATIONS, EVE_ITERATIONS

# Run with default parameters
main(
    num_epochs=ADV_ITERATIONS,
    num_alice_bob_iters=ALICE_BOB_ITERATIONS,
    num_eve_iters=EVE_ITERATIONS
)
```

### Using Trained Models for Signing and Verification

```python
import numpy as np
from neural_image_auth.inference import NeuralImageAuthenticator
from neural_image_auth.training.trainer import load_model
from neural_image_auth.crypto.key_manager import KeyManager

# Load trained models
alice = load_model("models/alice")
bob = load_model("models/bob")

# Load AES key
key_manager = KeyManager("keys")
aes_key = key_manager.load_key("default_key")

# Create authenticator
auth = NeuralImageAuthenticator(alice, bob, aes_key=aes_key)

# Sign an image
original_image = np.random.uniform(-1, 1, (64, 64, 3))
signed_image = auth.sign_image(original_image, message="SECRET123")

# Verify authenticity
result = auth.verify_image(signed_image)
print(f"Is Authentic: {result['is_authentic']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Extracted Message: {result['extracted_message']}")
print(f"Bit Error Rate: {result['bit_error_rate']:.2%}")
```

## Configuration

All hyperparameters are defined in `config.py`:

```python
# Image settings
IMAGE_SIZE = 64
CHANNELS = 3

# Message settings
MESSAGE_LENGTH = 256        # bits after AES encryption
RAW_MESSAGE_LENGTH = 128    # bits before encryption

# Training settings
BATCH_SIZE = 32
LEARNING_RATE = 0.0002
ADV_ITERATIONS = 50
ALICE_BOB_ITERATIONS = 20
EVE_ITERATIONS = 40

# Alice settings
PERTURBATION_SCALE = 0.05   # α - perturbation magnitude
PERTURBATION_BOUND = 0.1    # ε - max L∞ norm

# Loss weights
LAMBDA_RECONSTRUCTION = 1.0
LAMBDA_MESSAGE = 2.0
LAMBDA_AUTHENTICATION = 1.0
LAMBDA_IMPERCEPTIBILITY = 0.5
```

Modify these values before training to experiment with different configurations.

## Network Architectures

### Alice (U-Net Encoder-Decoder)

- **Input**: Image (64×64×3) + Message Channel (64×64×1)
- **Encoder**: 4 downsampling blocks (64→32→16→8 pixels)
- **Bottleneck**: 512 filters at 8×8
- **Decoder**: 3 upsampling blocks with skip connections
- **Output**: Perturbation map (64×64×3) → scaled and added to original

**Key Features**:
- Skip connections preserve spatial details
- Message injected early (concatenated with image)
- Residual perturbation for imperceptibility

### Bob (Shared CNN with Dual Heads)

- **Input**: Image (64×64×3)
- **Shared Encoder**: 4 downsampling blocks → 4×4×256 features
- **Head 1 (Message)**: Dense layers → Tanh output (message_length,)
- **Head 2 (Authentication)**: Dense layers with dropout → Sigmoid output

**Key Features**:
- Efficient shared representation learning
- Separate heads for different tasks
- Dropout for regularization in classification head

### Eve (Same Architecture as Alice)

- Separate U-Net for generating forged signatures
- Trained adversarially to fool Bob
- Attempts message extraction without keys

## Training Process

### Adversarial Training Loop

For each epoch:

1. **Phase 1 - Alice + Bob Cooperation** (20 iterations)
   - Alice embeds messages into images
   - Bob extracts messages and classifies as authentic
   - Combined loss optimizes both objectives

2. **Phase 2 - Bob Classifier** (20 iterations)
   - Bob trained on mixed batch (Alice-signed + unsigned)
   - Focus on authenticity classification

3. **Phase 3 - Eve Attacks** (40 iterations)
   - Eve attempts forgeries and message extraction
   - Bob frozen (not trained) during this phase

4. **Phase 4 - Bob Hardening** (10 iterations)
   - Bob trained to reject Eve's forged images
   - Distinguishes Alice-signed from Eve-forged images

### Loss Functions

- **Reconstruction Loss**: L2 distance between original and perturbed images
- **Message Extraction Loss**: L2 distance between original and extracted bits
- **Authentication Loss**: Binary cross-entropy for classification
- **Imperceptibility Loss**: Penalizes L∞ norm violations

## Evaluation Metrics

- **Bit Error Rate (BER)**: Percentage of incorrectly extracted bits
- **Message Accuracy**: 1 - BER
- **PSNR**: Peak Signal-to-Noise Ratio (image quality, target >40dB)
- **Authentication Accuracy**: Correct classification of authentic vs forged
- **Sensitivity/Specificity**: True positive/negative rates

### Expected Performance

- **Bit Extraction**: >95% accuracy on authentic images
- **Authentication**: >90% accuracy (accept signed, reject unsigned)
- **Eve Success Rate**: <20% (forgery attempts fail)
- **Imperceptibility**: PSNR >40dB

## Security Properties

| Property | Mechanism |
|----------|-----------|
| **Confidentiality** | AES-128-CBC encryption of watermark |
| **Authentication** | Neural signature verifiable only by trained Bob |
| **Integrity** | Message extraction fails if image tampered |
| **Non-repudiation** | Only holder of AES key + Alice weights can sign |

## API Reference

### NeuralImageAuthenticator

Main high-level API:

```python
auth = NeuralImageAuthenticator(alice_model, bob_model, aes_key)

# Sign an image
signed = auth.sign_image(image, message="SECRET")

# Verify authenticity
result = auth.verify_image(signed)
# Returns: {
#   'is_authentic': bool,
#   'confidence': float (0-1),
#   'extracted_message': str or None,
#   'bit_error_rate': float,
#   'extracted_bits': ndarray
# }

# Batch operations
signed_batch = auth.batch_sign_images(images, message="AUTH")
results = auth.batch_verify_images(images)
```

### AESCipher

Encryption utilities:

```python
cipher = AESCipher(key)

# Encrypt message to binary
bits = cipher.encrypt_to_bits("SECRET")  # → [-1, 1] array

# Decrypt from binary
message = cipher.decrypt_from_bits(bits)  # → "SECRET"

# Manual encryption
ciphertext, iv = cipher.encrypt(b"plaintext")
plaintext = cipher.decrypt(ciphertext, iv)
```

### KeyManager

Key storage management:

```python
manager = KeyManager("keys")

# Generate and save key
key = manager.generate_key(key_size=16)
path = manager.save_key(key, "my_key")

# Load existing key
key = manager.load_key("my_key")

# List and manage keys
keys = manager.list_keys()
manager.delete_key("my_key")
```

## Examples

### Example 1: Complete Training and Testing

```python
from neural_image_auth.main import main
from neural_image_auth.inference import NeuralImageAuthenticator
from neural_image_auth.crypto.key_manager import KeyManager
from neural_image_auth.utils import load_model
import numpy as np

# Train models
main(num_epochs=50)

# Load trained models and key
alice = load_model("logs/train_*/models/alice")
bob = load_model("logs/train_*/models/bob")
key_manager = KeyManager("keys")
aes_key = key_manager.load_key("default_key")

# Create authenticator
auth = NeuralImageAuthenticator(alice, bob, aes_key)

# Test signing and verification
test_image = np.random.uniform(-1, 1, (64, 64, 3))
signed = auth.sign_image(test_image, "SECURE_MSG")
result = auth.verify_image(signed)

assert result['is_authentic'] == True
assert result['extracted_message'] == "SECURE_MSG"
```

### Example 2: Robustness Testing

```python
from neural_image_auth.data.preprocessing import add_gaussian_noise, add_salt_and_pepper_noise

# Sign image
signed = auth.sign_image(image, "AUTH")

# Add noise (simulate tampering)
noisy = add_gaussian_noise(signed, std=0.01)
result = auth.verify_image(noisy)

# Decryption fails if tampering is too severe
if result['extracted_message'] is None:
    print("Image tampering detected!")
```

### Example 3: Batch Operations

```python
# Generate batch of images
batch_images = np.random.uniform(-1, 1, (32, 64, 64, 3))

# Sign all
signed_batch = auth.batch_sign_images(batch_images, "BATCH_AUTH")

# Verify all
results = auth.batch_verify_images(signed_batch)

# Analyze results
authentic_count = sum(1 for r in results if r['is_authentic'])
print(f"Authentic: {authentic_count}/{len(results)}")
```

## Testing

Run unit tests:

```bash
pytest tests/
```

## Troubleshooting

### Out of Memory Error
- Reduce `BATCH_SIZE` in `config.py`
- Reduce `IMAGE_SIZE` (note: affects network input)

### Poor Message Extraction
- Increase `LAMBDA_MESSAGE` in `config.py`
- Train for more epochs
- Check that `MESSAGE_LENGTH` matches between training and inference

### Eve Success Rate Too High
- Increase `EVE_ITERATIONS`
- Increase `LAMBDA_AUTHENTICATION` to focus Bob on classification
- Train for more epochs

## Performance Tips

1. **Use GPU**: TensorFlow automatically uses GPU if available
2. **Mixed Precision**: Enable in `config.py` for faster training
3. **Data Pipeline**: Precompute batches if possible
4. **Model Checkpointing**: Save models periodically during training

## References

- **Paper Concept**: Based on adversarial neural cryptography (similar to Abadi & Andersen, 2016)
- **AES Encryption**: NIST FIPS 197 standard
- **U-Net Architecture**: Ronneberger et al., 2015
- **Adversarial Training**: Goodfellow et al., 2014 (GANs)

## Authors

CS4379H Project Team - Cryptography Course

## License

This project is for educational purposes in the context of CS4379H Cryptography course.

## Disclaimer

This system is a **proof-of-concept** for educational purposes. While it demonstrates the feasibility of combining neural networks with cryptography:

- Not recommended for production use without security audit
- AES-CBC itself is secure, but neural network robustness is always an open question
- The adversarial training is simplified for pedagogical clarity

For production image authentication, consider industry standards like digital signatures or blockchain-based solutions.

---

**Questions?** Contact the course instructor or check the inline documentation in the code.
