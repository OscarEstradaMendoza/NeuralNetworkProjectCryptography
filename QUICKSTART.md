# Quick Start Guide - Neural Image Authentication System

## 5-Minute Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies:
- TensorFlow 2.10+ (deep learning)
- pycryptodome (AES encryption)
- numpy, Pillow (image processing)
- matplotlib (visualization)

### 2. Train the System (Optional - takes ~30 min on GPU)

```bash
cd /path/to/project
python -m neural_image_auth.main
```

This will:
- Initialize Alice, Bob, and Eve networks
- Generate AES key
- Run 50 epochs of adversarial training
- Save models to `logs/train_*/models/`
- Save training history to `logs/train_*/`

Or with custom parameters:

```python
from neural_image_auth.main import main

main(num_epochs=100, num_alice_bob_iters=30, num_eve_iters=50)
```

### 3. Quick Test

```python
import numpy as np
from neural_image_auth.inference import NeuralImageAuthenticator
from neural_image_auth.crypto.aes_cipher import AESCipher

# Initialize with trained models (or create new ones)
from neural_image_auth.models.alice import create_alice_network
from neural_image_auth.models.bob import create_bob_network

alice = create_alice_network()
bob = create_bob_network()
aes_key = b'sixteen_byte_key'  # 16 bytes for AES-128

# Create authenticator
auth = NeuralImageAuthenticator(alice, bob, aes_key)

# Create or load an image
test_image = np.random.uniform(-1, 1, (64, 64, 3))

# Sign the image
signed = auth.sign_image(test_image, message="SECRET_MSG")
print(f"✓ Image signed (shape: {signed.shape})")

# Verify authenticity
result = auth.verify_image(signed)
print(f"✓ Authentic: {result['is_authentic']}")
print(f"✓ Confidence: {result['confidence']:.2%}")
print(f"✓ Extracted Message: {result['extracted_message']}")
print(f"✓ Bit Error Rate: {result['bit_error_rate']:.2%}")
```

Expected output:
```
✓ Image signed (shape: (64, 64, 3))
✓ Authentic: True
✓ Confidence: 95.23%
✓ Extracted Message: SECRET_MSG
✓ Bit Error Rate: 2.34%
```

---

## Core Components

### 1. Signing Images

```python
from neural_image_auth.inference import NeuralImageAuthenticator

# Create authenticator
auth = NeuralImageAuthenticator(alice, bob, aes_key)

# Sign a single image
signed_image = auth.sign_image(
    image=original_image,
    message="AUTHENTIC"  # Default message
)

# Sign multiple images
signed_batch = auth.batch_sign_images(
    images=image_array,  # Shape: (batch_size, 64, 64, 3)
    message="BATCH_AUTH"
)
```

### 2. Verifying Images

```python
# Verify a single image
result = auth.verify_image(image)
# Returns: {'is_authentic': bool, 'confidence': float, 
#           'extracted_message': str, 'bit_error_rate': float}

# Verify multiple images
results = auth.batch_verify_images(image_batch)
```

### 3. Key Management

```python
from neural_image_auth.crypto.key_manager import KeyManager

key_mgr = KeyManager("keys/")

# Generate new key
aes_key = key_mgr.generate_key(key_size=16)  # AES-128
key_mgr.save_key(aes_key, "my_key")

# Load existing key
aes_key = key_mgr.load_key("my_key")

# List all keys
keys = key_mgr.list_keys()
```

### 4. Direct Encryption/Decryption

```python
from neural_image_auth.crypto.aes_cipher import AESCipher

cipher = AESCipher(aes_key)

# Encrypt message to bits
bits = cipher.encrypt_to_bits("SECRET")  # → [-1, 1] array (256 bits)

# Decrypt from bits
message = cipher.decrypt_from_bits(bits)  # → "SECRET"

# Manual encryption
ciphertext, iv = cipher.encrypt(b"plaintext")
plaintext = cipher.decrypt(ciphertext, iv)
```

### 5. Image Preprocessing

```python
from neural_image_auth.data.preprocessing import (
    preprocess_for_network,
    postprocess_from_network,
    add_gaussian_noise,
    normalize_image
)

# Preprocessing
normalized = normalize_image(image)  # [0, 255] → [-1, 1]

# Adding noise (for robustness testing)
noisy = add_gaussian_noise(image, std=0.01)

# Postprocessing
output = postprocess_from_network(image)  # [-1, 1] → [0, 255]
```

---

## Configuration

All hyperparameters are in `config.py`:

```python
# Image settings
IMAGE_SIZE = 64
CHANNELS = 3

# Message settings
MESSAGE_LENGTH = 256        # bits after AES encryption

# Training settings
BATCH_SIZE = 32
LEARNING_RATE = 0.0002

# Perturbation settings
PERTURBATION_SCALE = 0.05  # How much to perturb (α)
PERTURBATION_BOUND = 0.1   # Max L∞ norm (ε)

# Loss weights
LAMBDA_RECONSTRUCTION = 1.0      # Keep image close to original
LAMBDA_MESSAGE = 2.0             # Extract correct message
LAMBDA_AUTHENTICATION = 1.0      # Classify correctly
LAMBDA_IMPERCEPTIBILITY = 0.5    # Bound perturbation
```

Modify before training to experiment!

---

## Training from Scratch

### Step 1: Create Models

```python
from neural_image_auth.models.alice import create_alice_network
from neural_image_auth.models.bob import create_bob_network
from neural_image_auth.models.eve import create_eve_network

alice = create_alice_network()
bob = create_bob_network()
eve = create_eve_network()

print("Models created successfully!")
```

### Step 2: Initialize Trainer

```python
from neural_image_auth.training.trainer import AdversarialTrainer
from neural_image_auth.crypto.aes_cipher import AESCipher

aes_key = b'sixteen_byte_key'
trainer = AdversarialTrainer(alice, bob, eve, aes_key=aes_key)
```

### Step 3: Create Data Pipeline

```python
from neural_image_auth.data.datagen import DataPipeline
from neural_image_auth.config import BATCH_SIZE, IMAGE_SIZE, CHANNELS

pipeline = DataPipeline(
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    channels=CHANNELS
)
```

### Step 4: Run Training Loop

```python
# Simple single-epoch training
images = pipeline.get_training_batch()
message = "TEST"
message_bits = trainer.get_aes_cipher().encrypt_to_bits(message)

import tensorflow as tf
import numpy as np

message_bits = tf.constant(
    np.tile(message_bits[np.newaxis, :], [BATCH_SIZE, 1])
)

# Train Alice + Bob
loss, loss_dict, accuracy = trainer.train_step_alice_bob(images, message_bits)
print(f"Loss: {loss:.4f}, Bit Accuracy: {accuracy:.2%}")

# Train Eve
eve_loss = trainer.train_step_eve(images, message_bits)
print(f"Eve Loss: {eve_loss:.4f}")
```

### Step 5: Save Models

```python
from neural_image_auth.utils import save_model

save_model(alice, "alice", "models/")
save_model(bob, "bob", "models/")
save_model(eve, "eve", "models/")
```

---

## Testing and Evaluation

### 1. Check Authentication Accuracy

```python
from neural_image_auth.training.metrics import calculate_authentication_accuracy

# Get predictions on test batch
test_images = pipeline.get_test_batch()
_, predictions = bob(test_images, training=False)
predictions = predictions.numpy()

# Compare with ground truth
labels = np.ones(len(test_images))  # All should be signed
accuracy = calculate_authentication_accuracy(predictions, labels)
print(f"Authentication Accuracy: {accuracy:.2%}")
```

### 2. Check Message Extraction

```python
from neural_image_auth.training.metrics import calculate_ber

original_bits = trainer.get_aes_cipher().encrypt_to_bits("TEST")
# ... get extracted_bits from Bob ...
ber = calculate_ber(original_bits, extracted_bits)
print(f"Bit Error Rate: {ber:.2%}")
print(f"Message Accuracy: {1-ber:.2%}")
```

### 3. Check Image Quality

```python
from neural_image_auth.training.metrics import calculate_psnr

original = np.array(...)
perturbed = np.array(...)
psnr = calculate_psnr(original, perturbed)
print(f"PSNR: {psnr:.2f} dB (target >40)")
```

### 4. Robustness Testing

```python
from neural_image_auth.data.preprocessing import (
    add_gaussian_noise,
    add_salt_and_pepper_noise
)

# Create signed image
signed = auth.sign_image(image, "SECRET")

# Test with noise
noisy = add_gaussian_noise(signed, std=0.01)
result = auth.verify_image(noisy)
print(f"Under Gaussian noise: {result['is_authentic']}")

# Test with salt-and-pepper
noisy_sp = add_salt_and_pepper_noise(signed, salt_pepper_ratio=0.02)
result = auth.verify_image(noisy_sp)
print(f"Under S&P noise: {result['is_authentic']}")
```

---

## Common Issues & Solutions

### Issue: Out of Memory Error
**Solution**: Reduce BATCH_SIZE in config.py or IMAGE_SIZE

### Issue: Poor Message Extraction (BER > 10%)
**Solution**:
- Train for more epochs (increase ADV_ITERATIONS)
- Increase LAMBDA_MESSAGE loss weight
- Check that MESSAGE_LENGTH matches between training and inference

### Issue: Eve Success Rate Too High
**Solution**:
- Increase EVE_ITERATIONS in config.py
- Increase LAMBDA_AUTHENTICATION weight
- Train Bob hardening phase longer

### Issue: PSNR < 40dB (Too much visible perturbation)
**Solution**:
- Decrease PERTURBATION_SCALE (α)
- Increase LAMBDA_RECONSTRUCTION loss weight

### Issue: Models Won't Load
**Solution**:
```python
# Rebuild models with same architecture
from neural_image_auth.models.alice import create_alice_network
alice = create_alice_network()
alice.load_weights("models/alice")  # Load weights instead
```

---

## Performance Tips

1. **GPU Acceleration**: TensorFlow auto-detects GPU
   ```python
   import tensorflow as tf
   gpus = tf.config.list_physical_devices('GPU')
   print(f"GPUs available: {len(gpus)}")
   ```

2. **Mixed Precision Training** (faster on modern GPUs):
   ```python
   from tensorflow.keras import mixed_precision
   policy = mixed_precision.Policy('mixed_float16')
   mixed_precision.set_global_policy(policy)
   ```

3. **Batch Processing**: Use batch operations for speed
   ```python
   signed_batch = auth.batch_sign_images(images)
   results = auth.batch_verify_images(images)
   ```

---

## Next Steps

1. **Understand the architecture**: Read code comments in `models/`
2. **Study the training loop**: Follow `training/trainer.py`
3. **Try modifications**: Change hyperparameters in `config.py`
4. **Experiment**: Test robustness with different noise levels
5. **Extend**: Add new loss functions, metrics, or architectures

---

## Documentation

- **README_NEW.md**: Comprehensive documentation
- **IMPLEMENTATION_SUMMARY.md**: Implementation details
- **config.py**: All hyperparameters documented
- **Inline comments**: Throughout the code

## Support

For questions about specific components:
- Cryptography: See `crypto/aes_cipher.py`
- Models: See `models/alice.py`, `models/bob.py`
- Training: See `training/trainer.py`
- Inference: See `inference.py`

Happy experimenting! 🚀
