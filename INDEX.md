# 📖 Index - Neural Image Authentication System

## Quick Navigation Guide

### 🚀 Getting Started (5 minutes)
1. **First time?** → Start with `QUICKSTART.md`
2. **Install packages** → `pip install -r requirements.txt`
3. **Try a quick test** → See "Quick Test" section in `QUICKSTART.md`

### 📚 Understanding the System
1. **System overview** → `README_NEW.md` (Section: System Overview)
2. **Architecture** → `ARCHITECTURE.md` (Visual diagrams)
3. **How it works** → `README_NEW.md` (Section: Signing/Verification Flow)

### 🔍 Exploring the Code

#### Cryptography
- **Where?** `neural_image_auth/crypto/`
- **Files**: `aes_cipher.py`, `key_manager.py`
- **Learn**: AES encryption, key management
- **Example**: See "Encryption/Decryption" in `QUICKSTART.md`

#### Neural Networks
- **Where?** `neural_image_auth/models/`
- **Files**: `alice.py`, `bob.py`, `eve.py`
- **Learn**: U-Net, CNN, adversarial networks
- **Example**: See "Network Architectures" in `ARCHITECTURE.md`

#### Training
- **Where?** `neural_image_auth/training/`
- **Files**: `losses.py`, `trainer.py`, `metrics.py`
- **Learn**: Loss functions, training loop
- **Example**: See "Training from Scratch" in `QUICKSTART.md`

#### Data & Preprocessing
- **Where?** `neural_image_auth/data/`
- **Files**: `datagen.py`, `preprocessing.py`
- **Learn**: Image generation, preprocessing
- **Example**: See "Image Preprocessing" in `QUICKSTART.md`

#### Inference
- **Where?** `neural_image_auth/inference.py`
- **Learn**: Signing, verification, API usage
- **Example**: See "Core Components" in `QUICKSTART.md`

### 📊 Configuration & Customization
- **Configuration file**: `neural_image_auth/config.py`
- **What can I change?** → See `config.py` comments
- **Performance tips?** → `QUICKSTART.md` (Performance Tips)

### 🧪 Testing & Evaluation

#### Unit Tests
- **Where?** `TESTING.md` (Section: Unit Testing Examples)
- **What to test?** Cryptography, preprocessing, networks, losses
- **How to run?** Copy example code and run

#### Integration Tests
- **Where?** `TESTING.md` (Section: Integration Tests)
- **What to test?** Complete signing/verification workflow
- **How to run?** Copy example code and run

#### Robustness Tests
- **Where?** `TESTING.md` (Section: Robustness Tests)
- **What to test?** Noise, compression, tampering
- **How to run?** Copy example code and run

#### Performance Tests
- **Where?** `TESTING.md` (Section: Performance Tests)
- **What to measure?** Training speed, model size
- **How to run?** Copy example code and run

### 🎯 Common Tasks

#### Task: Train the System
1. Go to: `QUICKSTART.md` (Train the System section)
2. Or: Run `python -m neural_image_auth.main`
3. Results saved to: `logs/train_TIMESTAMP/`

#### Task: Sign an Image
1. Code example: `QUICKSTART.md` (Using Trained Models)
2. Or: `README_NEW.md` (API Reference: NeuralImageAuthenticator)

#### Task: Verify an Image
1. Code example: `QUICKSTART.md` (Using Trained Models)
2. Or: `README_NEW.md` (API Reference: verify_image)

#### Task: Test Robustness
1. Guide: `QUICKSTART.md` (Robustness Testing)
2. Or: `TESTING.md` (Test Robustness to Noise)

#### Task: Change Hyperparameters
1. Edit: `neural_image_auth/config.py`
2. Modify: Any LAMBDA_* or other constants
3. Run: Training with new config

#### Task: Understand Losses
1. Read: `neural_image_auth/training/losses.py`
2. Visualize: `ARCHITECTURE.md` (Loss Function Computation)
3. Formula: `README_NEW.md` (Expected Behavior section)

### 📖 Full Documentation Structure

```
📄 QUICKSTART.md
   └─ For first-time users, code examples

📄 README_NEW.md
   └─ Complete reference, system overview, API docs

📄 ARCHITECTURE.md
   └─ ASCII diagrams, visual explanations

📄 IMPLEMENTATION_SUMMARY.md
   └─ Technical details, code statistics

📄 TESTING.md
   └─ Test examples, testing guide

📄 PROJECT_COMPLETION.md
   └─ Final summary, statistics, achievements

📄 requirements.txt
   └─ Package dependencies
```

### 🔗 File-to-Concept Mapping

| Concept | Files |
|---------|-------|
| AES Encryption | `crypto/aes_cipher.py` |
| Key Management | `crypto/key_manager.py` |
| Image Generation | `data/datagen.py` |
| Preprocessing | `data/preprocessing.py` |
| Alice Network | `models/alice.py` |
| Bob Network | `models/bob.py` |
| Eve Network | `models/eve.py` |
| Loss Functions | `training/losses.py` |
| Training Loop | `training/trainer.py` |
| Metrics | `training/metrics.py` |
| Inference API | `inference.py` |
| Configuration | `config.py` |
| Entry Point | `main.py` |
| Utilities | `utils.py` |

### 💡 Learning Path

#### Beginner (Understand the system)
1. Read: `QUICKSTART.md` (overview)
2. Read: `ARCHITECTURE.md` (diagrams)
3. Read: `README_NEW.md` (complete overview)

#### Intermediate (Understand components)
1. Read: `config.py` (parameters)
2. Read: `neural_image_auth/crypto/aes_cipher.py` (encryption)
3. Read: `neural_image_auth/models/alice.py` (architecture)
4. Read: `neural_image_auth/training/losses.py` (training)

#### Advanced (Understand implementation details)
1. Study: All files in `neural_image_auth/`
2. Review: `IMPLEMENTATION_SUMMARY.md`
3. Run: Testing examples from `TESTING.md`
4. Experiment: Modify `config.py` and retrain

### ❓ Frequently Asked Questions

**Q: Where do I start?**
A: Read `QUICKSTART.md` → Try the quick test → Explore code

**Q: How do I train?**
A: Run `python -m neural_image_auth.main` (see `QUICKSTART.md`)

**Q: How do I sign an image?**
A: See "Using Trained Models" in `QUICKSTART.md`

**Q: What if I get an error?**
A: See "Common Issues & Solutions" in `QUICKSTART.md`

**Q: Can I change hyperparameters?**
A: Yes, edit `neural_image_auth/config.py`

**Q: How does the encryption work?**
A: Read `crypto/aes_cipher.py` and `README_NEW.md` (Security section)

**Q: How do the networks work?**
A: See `ARCHITECTURE.md` (Network Architectures)

**Q: What's the expected performance?**
A: See `README_NEW.md` (Expected Behavior) or `PROJECT_COMPLETION.md`

### 📊 Project Statistics at a Glance

- **Total Python Code**: 4,312 lines
- **Total Documentation**: 2,500+ lines
- **Python Files**: 19
- **Modules**: 13
- **Classes**: 8 major
- **Functions**: 50+
- **Test Examples**: 10+

### ✅ Verification Checklist

Before using the system:
- [ ] Downloaded `requirements.txt`
- [ ] Installed with `pip install -r requirements.txt`
- [ ] Read `QUICKSTART.md`
- [ ] Understood `ARCHITECTURE.md`
- [ ] Reviewed `README_NEW.md`

### 🚀 Next Steps

1. **Quick Start**: Run the quick test in `QUICKSTART.md`
2. **Train**: Run `python -m neural_image_auth.main`
3. **Test**: Run examples from `TESTING.md`
4. **Customize**: Modify `config.py` and experiment
5. **Deploy**: Use inference API from `inference.py`

### 💬 Key Concepts

**AES-CBC Encryption**: Secure encryption mode for message confidentiality

**U-Net Architecture**: Image-to-image network with skip connections

**Adversarial Training**: Alice/Bob vs Eve (attacker)

**Imperceptibility**: Perturbation invisible to human eye (PSNR >40 dB)

**Message Extraction**: Bob recovers encrypted message from signed image

**Authenticity Classification**: Bob verifies image is from Alice

### 📞 Support

For specific topics:
- **Cryptography questions** → `crypto/aes_cipher.py` + `README_NEW.md`
- **Architecture questions** → `ARCHITECTURE.md`
- **Training questions** → `training/trainer.py` + `QUICKSTART.md`
- **API questions** → `inference.py` + `README_NEW.md`
- **Testing questions** → `TESTING.md`

### 🎓 Course Context

This is a **CS4379H Cryptography Course** project demonstrating:
- Classical cryptography (AES)
- Modern deep learning (neural networks)
- Adversarial machine learning
- Image processing and security
- Software engineering best practices

---

**Happy Learning!** 🚀

Start with `QUICKSTART.md` and explore from there.
