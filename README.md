<<<<<<< HEAD
# NeuralNetworkProjectCryptography
A neural cryptography system combining AES-128-CBC encryption with adversarial neural networks for image authentication. Alice embeds encrypted watermarks as imperceptible perturbations, Bob extracts and verifies them, and Eve attempts forgery to strengthen the system through adversarial training.
=======
# Adversarial-Neural-Cryptography
An implementation of the ICLR 2017 paper 'Learning to Protect Communications
with Adversarial Neural Cryptography'.

## Dependencies
Simply run:
```bash
pip install -r requirements.txt
```

## Running the script
In order to start training the agents: Alice, Bob and Eve run:
```bash
python3 neural_encryption.py
```

## To visualize the training graph during the run:
```bash
tensorboard --logdir logs
```
>>>>>>> d3d99d4 (Initial)
