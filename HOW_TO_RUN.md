# How to Run This Project

## Quick Start (Recommended Method)

### Step 1: Create Virtual Environment (Required)

**Important**: This project requires a virtual environment to avoid Python package conflicts.

```bash
cd "/Users/oscarestradamendoza/Desktop/Cryptography_Project"

# Create virtual environment (if not already created)
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

### Step 2: Install Dependencies

```bash
# Make sure virtual environment is activated (you should see (venv) in your prompt)
pip install -r requirements.txt
```

### Step 3: Run the Main Training Script

**Option A: Using Python module (Recommended)**
```bash
python -m neural_image_auth.main
```

**Option B: Using Python script directly**
```bash
python neural_image_auth/main.py
```

**Option C: Custom parameters**
```python
from neural_image_auth.main import main

# Run with custom parameters
main(
    num_epochs=50,           # Number of adversarial training epochs
    num_alice_bob_iters=20,  # Alice+Bob training iterations per epoch
    num_eve_iters=40         # Eve training iterations per epoch
)
```

**Important**: Always activate the virtual environment before running:
```bash
source venv/bin/activate  # On macOS/Linux
```

### Step 4: Monitor Training (Optional)

In a separate terminal, run TensorBoard to visualize training:
```bash
tensorboard --logdir logs
```
Then open http://localhost:6006 in your browser.

---

## Alternative: Original Neural Encryption Script

If you want to run the original TensorFlow 1.x implementation:

```bash
cd "/Users/oscarestradamendoza/Desktop/Cryptography_Project"
python3 neural_encryption.py
```

**Note:** This uses TensorFlow 1.x style code and may require TensorFlow 1.x to be installed.

---

## Using Trained Models for Inference

After training, you can use the models to sign and verify images:

```python
import numpy as np
from neural_image_auth.inference import NeuralImageAuthenticator
from neural_image_auth.utils import load_model
from neural_image_auth.crypto.key_manager import KeyManager

# Load trained models (adjust path based on your training run)
alice = load_model("logs/train_*/models/alice")
bob = load_model("logs/train_*/models/bob")

# Load AES key
key_manager = KeyManager("keys")
aes_key = key_manager.load_key("default_key")

# Create authenticator
auth = NeuralImageAuthenticator(alice, bob, aes_key=aes_key)

# Sign an image
test_image = np.random.uniform(-1, 1, (64, 64, 3))
signed_image = auth.sign_image(test_image, message="SECRET123")

# Verify authenticity
result = auth.verify_image(signed_image)
print(f"Is Authentic: {result['is_authentic']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Extracted Message: {result['extracted_message']}")
print(f"Bit Error Rate: {result['bit_error_rate']:.2%}")
```

---

## Quick Test (Without Full Training)

To quickly test the system without full training:

```python
import numpy as np
from neural_image_auth.inference import NeuralImageAuthenticator
from neural_image_auth.models.alice import create_alice_network
from neural_image_auth.models.bob import create_bob_network
from neural_image_auth.crypto.aes_cipher import AESCipher

# Create models (untrained, for testing)
alice = create_alice_network()
bob = create_bob_network()
aes_key = b'sixteen_byte_key'  # 16 bytes for AES-128

# Create authenticator
auth = NeuralImageAuthenticator(alice, bob, aes_key)

# Test with a random image
test_image = np.random.uniform(-1, 1, (64, 64, 3))
signed = auth.sign_image(test_image, message="TEST_MSG")
result = auth.verify_image(signed)

print(f"Authentic: {result['is_authentic']}")
print(f"Extracted Message: {result['extracted_message']}")
```

---

## Configuration

All hyperparameters can be modified in `neural_image_auth/config.py`:

- `IMAGE_SIZE = 64` - Image dimensions
- `BATCH_SIZE = 32` - Training batch size
- `ADV_ITERATIONS = 50` - Number of training epochs
- `LEARNING_RATE = 0.0002` - Learning rate
- And many more...

---

## Expected Training Time

- **CPU**: ~2-4 hours for 50 epochs
- **GPU**: ~30 minutes for 50 epochs

---

## Troubleshooting

### Issue: ModuleNotFoundError
**Solution**: Make sure you've activated the virtual environment and installed dependencies:
```bash
source venv/bin/activate  # Activate virtual environment
pip install -r requirements.txt  # Install dependencies
```

### Issue: "externally-managed-environment" error
**Solution**: This means you're trying to install packages system-wide. Use a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: Out of Memory
**Solution**: Reduce `BATCH_SIZE` in `neural_image_auth/config.py`

### Issue: TensorFlow version conflicts
**Solution**: The project requires TensorFlow 2.10+. If you have issues:
```bash
pip install --upgrade tensorflow>=2.10.0
```

---

## Project Structure

- `neural_image_auth/main.py` - Main entry point for training
- `neural_image_auth/inference.py` - Inference API for signing/verifying
- `neural_image_auth/config.py` - All configuration parameters
- `neural_encryption.py` - Original TensorFlow 1.x implementation
- `requirements.txt` - Python dependencies

---

## More Information

- See `QUICKSTART.md` for detailed usage examples
- See `README_NEW.md` for comprehensive documentation
- See `README.md` for basic overview

---

## Running the GUI Applications

You have **two GUI options** in this project:

- **Desktop GUI (Tkinter)**: `gui_app.py`
- **Web GUI (Flask in browser)**: `web_gui.py`

### Desktop GUI (Tkinter)

```bash
cd "/Users/oscarestradamendoza/Desktop/Cryptography_Project"
source venv/bin/activate
python gui_app.py
```

This opens a native desktop window for signing and verifying images.

### Web GUI (Browser-based)

```bash
cd "/Users/oscarestradamendoza/Desktop/Cryptography_Project"
source venv/bin/activate
python web_gui.py
```

Then open the printed `http://localhost:[PORT]` URL in your browser. You can also use:

```bash
./start_gui.sh
```

to automatically activate the virtual environment and start the web GUI from the project root.

