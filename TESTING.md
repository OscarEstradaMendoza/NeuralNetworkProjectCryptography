# Testing Guide - Neural Image Authentication System

## Unit Testing Examples

### 1. Test AES Cipher

```python
from neural_image_auth.crypto.aes_cipher import AESCipher
import numpy as np

def test_aes_cipher():
    """Test AES encryption and decryption."""
    cipher = AESCipher()
    
    # Test 1: Message round-trip
    message = "SECRET_MESSAGE_123"
    encrypted_bits = cipher.encrypt_to_bits(message)
    decrypted = cipher.decrypt_from_bits(encrypted_bits)
    
    assert decrypted == message, f"Expected {message}, got {decrypted}"
    print("✓ AES cipher round-trip test passed")
    
    # Test 2: Bit format
    assert encrypted_bits.dtype == np.float32
    assert encrypted_bits.min() >= -1.0 and encrypted_bits.max() <= 1.0
    print("✓ Bit format test passed")
    
    # Test 3: Different keys
    cipher2 = AESCipher()  # Different random key
    try:
        decrypted2 = cipher2.decrypt_from_bits(encrypted_bits)
        print("✗ Should fail with wrong key!")
    except:
        print("✓ Wrong key decryption test passed")

test_aes_cipher()
```

### 2. Test Image Preprocessing

```python
from neural_image_auth.data.preprocessing import (
    normalize_image, denormalize_image, 
    preprocess_for_network, postprocess_from_network
)
import numpy as np

def test_preprocessing():
    """Test image preprocessing pipeline."""
    
    # Test 1: Normalization round-trip
    original = np.random.uniform(0, 1, (64, 64, 3))
    normalized = normalize_image(original)
    denormalized = denormalize_image(normalized)
    
    assert np.allclose(original, denormalized), "Normalization round-trip failed"
    print("✓ Normalization round-trip test passed")
    
    # Test 2: Range checking
    assert normalized.min() >= -1.0 and normalized.max() <= 1.0
    assert denormalized.min() >= 0.0 and denormalized.max() <= 1.0
    print("✓ Range checking test passed")
    
    # Test 3: Full preprocessing pipeline
    raw_image = np.random.uniform(0, 255, (128, 128, 3))
    preprocessed = preprocess_for_network(raw_image, target_size=64)
    
    assert preprocessed.shape == (64, 64, 3)
    assert preprocessed.min() >= -1.0 and preprocessed.max() <= 1.0
    print("✓ Full preprocessing pipeline test passed")

test_preprocessing()
```

### 3. Test Alice Network

```python
from neural_image_auth.models.alice import create_alice_network
from neural_image_auth.config import IMAGE_SIZE, CHANNELS, MESSAGE_LENGTH, BATCH_SIZE
import tensorflow as tf
import numpy as np

def test_alice_network():
    """Test Alice encoder network."""
    
    alice = create_alice_network()
    
    # Test 1: Forward pass
    batch_images = tf.constant(np.random.uniform(-1, 1, (BATCH_SIZE, IMAGE_SIZE, CHANNELS, 3)))
    batch_bits = tf.constant(np.random.uniform(-1, 1, (BATCH_SIZE, MESSAGE_LENGTH)))
    
    output = alice([batch_images, batch_bits], training=False)
    
    assert output.shape == batch_images.shape, "Output shape mismatch"
    assert output.dtype == tf.float32
    print("✓ Alice forward pass test passed")
    
    # Test 2: Output range
    assert tf.reduce_min(output) >= -1.0 and tf.reduce_max(output) <= 1.0
    print("✓ Alice output range test passed")
    
    # Test 3: Imperceptibility
    perturbation = tf.abs(output - batch_images)
    max_perturbation = tf.reduce_max(perturbation)
    
    assert max_perturbation < 0.2, "Perturbation too large"
    print("✓ Alice imperceptibility test passed")

test_alice_network()
```

### 4. Test Bob Network

```python
from neural_image_auth.models.bob import create_bob_network
from neural_image_auth.config import IMAGE_SIZE, CHANNELS, MESSAGE_LENGTH, BATCH_SIZE
import tensorflow as tf
import numpy as np

def test_bob_network():
    """Test Bob decoder/classifier network."""
    
    bob = create_bob_network()
    
    # Test 1: Dual outputs
    batch_images = tf.constant(np.random.uniform(-1, 1, (BATCH_SIZE, IMAGE_SIZE, CHANNELS, 3)))
    
    extracted_bits, auth_prob = bob(batch_images, training=False)
    
    assert extracted_bits.shape == (BATCH_SIZE, MESSAGE_LENGTH)
    assert auth_prob.shape == (BATCH_SIZE, 1)
    print("✓ Bob dual output test passed")
    
    # Test 2: Output ranges
    assert tf.reduce_min(extracted_bits) >= -1.0 and tf.reduce_max(extracted_bits) <= 1.0
    assert tf.reduce_min(auth_prob) >= 0.0 and tf.reduce_max(auth_prob) <= 1.0
    print("✓ Bob output range test passed")
    
    # Test 3: Gradient flow
    with tf.GradientTape() as tape:
        extracted_bits, auth_prob = bob(batch_images, training=True)
        loss = tf.reduce_mean(extracted_bits) + tf.reduce_mean(auth_prob)
    
    gradients = tape.gradient(loss, bob.trainable_variables)
    assert all(g is not None for g in gradients), "Gradient flow broken"
    print("✓ Bob gradient flow test passed")

test_bob_network()
```

### 5. Test Loss Functions

```python
from neural_image_auth.training.losses import (
    reconstruction_loss, message_extraction_loss, 
    bit_accuracy, alice_bob_combined_loss
)
from neural_image_auth.config import LAMBDAS
import tensorflow as tf
import numpy as np

def test_losses():
    """Test loss functions."""
    
    batch_size = 32
    img_size = 64
    msg_length = 256
    
    # Test 1: Reconstruction loss
    original = tf.constant(np.random.uniform(-1, 1, (batch_size, img_size, img_size, 3)))
    perturbed = original + 0.01 * tf.random.normal(original.shape)
    
    loss = reconstruction_loss(original, perturbed)
    assert loss > 0 and loss < 0.1, "Reconstruction loss out of range"
    print("✓ Reconstruction loss test passed")
    
    # Test 2: Message extraction loss
    original_bits = tf.constant(np.random.uniform(-1, 1, (batch_size, msg_length)))
    extracted_bits = original_bits + 0.05 * tf.random.normal(original_bits.shape)
    
    loss = message_extraction_loss(original_bits, extracted_bits)
    assert loss > 0, "Message extraction loss should be positive"
    print("✓ Message extraction loss test passed")
    
    # Test 3: Bit accuracy
    accuracy = bit_accuracy(original_bits, original_bits)  # Perfect extraction
    assert accuracy.numpy() == 1.0, "Perfect extraction should have 100% accuracy"
    print("✓ Bit accuracy test passed")
    
    # Test 4: Combined loss
    auth_pred = tf.constant(np.random.uniform(0, 1, (batch_size, 1)))
    
    total_loss, loss_dict = alice_bob_combined_loss(
        original, perturbed, original_bits, extracted_bits,
        auth_pred, LAMBDAS
    )
    
    assert total_loss > 0, "Total loss should be positive"
    assert len(loss_dict) == 4, "Should have 4 loss components"
    print("✓ Combined loss test passed")

test_losses()
```

## Integration Tests

### 6. Test Complete Signing and Verification

```python
from neural_image_auth.inference import NeuralImageAuthenticator
from neural_image_auth.models.alice import create_alice_network
from neural_image_auth.models.bob import create_bob_network
from neural_image_auth.crypto.aes_cipher import AESCipher
import numpy as np

def test_signing_verification():
    """Test complete signing and verification workflow."""
    
    # Initialize models and authenticator
    alice = create_alice_network()
    bob = create_bob_network()
    aes_key = b'sixteen_byte_key'  # 16 bytes
    
    auth = NeuralImageAuthenticator(alice, bob, aes_key)
    
    # Create test image
    test_image = np.random.uniform(-1, 1, (64, 64, 3))
    message = "TEST_MESSAGE"
    
    # Test 1: Sign image
    signed = auth.sign_image(test_image, message)
    assert signed.shape == (64, 64, 3)
    assert signed.dtype == np.uint8
    assert 0 <= signed.min() and signed.max() <= 255
    print("✓ Image signing test passed")
    
    # Test 2: Verify authentic image
    result = auth.verify_image(signed)
    assert 'is_authentic' in result
    assert 'confidence' in result
    assert 'extracted_message' in result
    assert 'bit_error_rate' in result
    print("✓ Image verification test passed")
    
    # Test 3: Check result values
    assert isinstance(result['is_authentic'], (bool, np.bool_))
    assert 0 <= result['confidence'] <= 1
    assert 0 <= result['bit_error_rate'] <= 1
    print("✓ Result format test passed")

test_signing_verification()
```

### 7. Test Batch Operations

```python
from neural_image_auth.inference import NeuralImageAuthenticator
from neural_image_auth.models.alice import create_alice_network
from neural_image_auth.models.bob import create_bob_network
import numpy as np

def test_batch_operations():
    """Test batch signing and verification."""
    
    alice = create_alice_network()
    bob = create_bob_network()
    aes_key = b'sixteen_byte_key'
    
    auth = NeuralImageAuthenticator(alice, bob, aes_key)
    
    # Create batch of images
    batch_size = 8
    batch_images = np.random.uniform(-1, 1, (batch_size, 64, 64, 3))
    
    # Test 1: Batch signing
    signed_batch = auth.batch_sign_images(batch_images, "BATCH_AUTH")
    assert signed_batch.shape == (batch_size, 64, 64, 3)
    print("✓ Batch signing test passed")
    
    # Test 2: Batch verification
    results = auth.batch_verify_images(signed_batch)
    assert len(results) == batch_size
    assert all('is_authentic' in r for r in results)
    print("✓ Batch verification test passed")

test_batch_operations()
```

## Robustness Tests

### 8. Test Robustness to Noise

```python
from neural_image_auth.inference import NeuralImageAuthenticator
from neural_image_auth.data.preprocessing import add_gaussian_noise, add_salt_and_pepper_noise
from neural_image_auth.models.alice import create_alice_network
from neural_image_auth.models.bob import create_bob_network
import numpy as np

def test_robustness():
    """Test robustness to various attacks."""
    
    alice = create_alice_network()
    bob = create_bob_network()
    aes_key = b'sixteen_byte_key'
    
    auth = NeuralImageAuthenticator(alice, bob, aes_key)
    
    # Create and sign image
    test_image = np.random.uniform(-1, 1, (64, 64, 3))
    signed = auth.sign_image(test_image, "SECRET")
    
    # Test 1: Gaussian noise
    noise_levels = [0.001, 0.005, 0.01]
    for std in noise_levels:
        noisy = add_gaussian_noise(signed / 255, std=std)
        result = auth.verify_image(noisy)
        print(f"  Gaussian noise (σ={std}): Auth={result['is_authentic']}, "
              f"BER={result['bit_error_rate']:.2%}")
    
    # Test 2: Salt-and-pepper noise
    noise_ratios = [0.005, 0.01, 0.02]
    for ratio in noise_ratios:
        noisy = add_salt_and_pepper_noise(signed / 255, salt_pepper_ratio=ratio)
        result = auth.verify_image(noisy)
        print(f"  S&P noise (ratio={ratio}): Auth={result['is_authentic']}, "
              f"BER={result['bit_error_rate']:.2%}")
    
    print("✓ Robustness test completed")

test_robustness()
```

## Performance Tests

### 9. Test Training Speed

```python
from neural_image_auth.training.trainer import AdversarialTrainer
from neural_image_auth.models.alice import create_alice_network
from neural_image_auth.models.bob import create_bob_network
from neural_image_auth.models.eve import create_eve_network
from neural_image_auth.data.datagen import DataPipeline
from neural_image_auth.config import BATCH_SIZE, MESSAGE_LENGTH
import tensorflow as tf
import numpy as np
import time

def test_training_speed():
    """Test training speed."""
    
    # Initialize
    alice = create_alice_network()
    bob = create_bob_network()
    eve = create_eve_network()
    trainer = AdversarialTrainer(alice, bob, eve)
    pipeline = DataPipeline(batch_size=BATCH_SIZE)
    
    # Get batch
    images = pipeline.get_training_batch()
    message = "TEST"
    message_bits = trainer.get_aes_cipher().encrypt_to_bits(message)
    message_bits = tf.constant(
        np.tile(message_bits[np.newaxis, :], [BATCH_SIZE, 1])
    )
    
    # Warmup
    trainer.train_step_alice_bob(images, message_bits)
    
    # Time 10 iterations
    start = time.time()
    for _ in range(10):
        images = pipeline.get_training_batch()
        trainer.train_step_alice_bob(images, message_bits)
    elapsed = time.time() - start
    
    per_iter = elapsed / 10
    print(f"✓ Average time per iteration: {per_iter:.3f}s")
    print(f"✓ Estimated time per epoch (100 iterations): {per_iter * 100:.1f}s")

test_training_speed()
```

### 10. Test Model Size

```python
from neural_image_auth.utils import calculate_model_size, print_model_summary
from neural_image_auth.models.alice import create_alice_network
from neural_image_auth.models.bob import create_bob_network
from neural_image_auth.models.eve import create_eve_network

def test_model_sizes():
    """Check model parameter counts."""
    
    alice = create_alice_network()
    bob = create_bob_network()
    eve = create_eve_network()
    
    for model, name in [(alice, "Alice"), (bob, "Bob"), (eve, "Eve")]:
        print_model_summary(model, name)
        sizes = calculate_model_size(model)
        print(f"{name}: {sizes['total_mb']} MB")

test_model_sizes()
```

## Automated Test Suite

```python
# test_suite.py
import unittest
from neural_image_auth.crypto.aes_cipher import AESCipher
from neural_image_auth.models.alice import create_alice_network
from neural_image_auth.models.bob import create_bob_network

class TestAESCipher(unittest.TestCase):
    def test_round_trip(self):
        cipher = AESCipher()
        message = "TEST_MESSAGE"
        bits = cipher.encrypt_to_bits(message)
        decrypted = cipher.decrypt_from_bits(bits)
        self.assertEqual(message, decrypted)

class TestAliceNetwork(unittest.TestCase):
    def test_forward_pass(self):
        import tensorflow as tf
        import numpy as np
        from neural_image_auth.config import BATCH_SIZE, MESSAGE_LENGTH
        
        alice = create_alice_network()
        images = tf.constant(np.random.uniform(-1, 1, (BATCH_SIZE, 64, 64, 3)))
        bits = tf.constant(np.random.uniform(-1, 1, (BATCH_SIZE, MESSAGE_LENGTH)))
        
        output = alice([images, bits], training=False)
        self.assertEqual(output.shape, images.shape)

if __name__ == '__main__':
    unittest.main()
```

Run with:
```bash
python -m pytest test_suite.py -v
```

---

## Continuous Integration Checklist

- [ ] All unit tests pass
- [ ] Integration tests pass
- [ ] Robustness tests pass
- [ ] No memory leaks detected
- [ ] Training converges properly
- [ ] PSNR > 40 dB
- [ ] Authentication accuracy > 90%
- [ ] Eve success rate < 20%
- [ ] Documentation up to date
- [ ] Code follows style guidelines

## Additional Testing Resources

- **Test Data**: Generate with `DataPipeline`
- **Metrics**: Use `training.metrics` module
- **Visualization**: Use `utils.visualize_signed_images()`
- **Logging**: Check `logs/` directory for training curves
