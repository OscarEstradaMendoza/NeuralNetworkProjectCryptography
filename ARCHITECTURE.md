# Neural Image Authentication System - Architecture Diagrams

## 1. System-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   NEURAL IMAGE AUTHENTICATION                   │
│                                                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │   CONFIG     │      │   CRYPTO     │      │    DATA      │  │
│  ├──────────────┤      ├──────────────┤      ├──────────────┤  │
│  │ • Image Size │      │ • AES-CBC    │      │ • Image Gen  │  │
│  │ • Batch Size │      │ • Key Mgmt   │      │ • Preprocess │  │
│  │ • Loss Wts   │      │ • Bit Conv   │      │ • Noise      │  │
│  │ • LR/Epochs  │      │              │      │              │  │
│  └──────────────┘      └──────────────┘      └──────────────┘  │
│                                                                   │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │   MODELS     │      │   TRAINING   │      │ INFERENCE    │  │
│  ├──────────────┤      ├──────────────┤      ├──────────────┤  │
│  │ • Alice      │      │ • Losses     │      │ • Sign Img   │  │
│  │ • Bob        │      │ • Trainer    │      │ • Verify     │  │
│  │ • Eve        │      │ • Metrics    │      │ • Batch Ops  │  │
│  │              │      │              │      │              │  │
│  └──────────────┘      └──────────────┘      └──────────────┘  │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              UTILITIES & MAIN ENTRY POINT                │    │
│  │  • Model Save/Load • Visualization • Training History   │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## 2. Signing Flow

```
                    SIGNING WORKFLOW

    Original Image                Secret Message
         (64×64×3)                  ("SECRET")
            │                            │
            │                            │
            ▼                            ▼
         ┌──────────┐              ┌──────────┐
         │ Preprocess│             │ AES Cipher
         │           │             │ (encrypt)
         └──────────┘              └──────────┘
            │                            │
            │                ┌───────────┘
            │                │
            ▼                ▼
         ┌──────────────────────────┐
         │   ALICE NETWORK          │
         │  (U-Net Encoder-Decoder) │
         ├──────────────────────────┤
         │ Input: Image + Message   │
         │ Processing: Embed        │
         │ Output: Perturbation     │
         └──────────────────────────┘
            │
            │ Signed Image
            │ (imperceptible watermark)
            │
            ▼
         ┌──────────┐
         │Postprocess
         └──────────┘
            │
            ▼
       SIGNED IMAGE
     [0, 255] uint8
```

## 3. Alice Network Architecture (U-Net)

```
                           ALICE (ENCODER-DECODER)

Input: Image (64×64×3) + Message (64×64×1) = Combined (64×64×4)

                            ┌─────────────┐
                            │   Input     │
                            │ (64×64×4)   │
                            └──────┬──────┘
                                   │
        ┌──────────────────────────┴──────────────────────────┐
        │                ENCODER PATH                         │
        │
        ├──► Conv(64) → BN → LeakyReLU ─────┐    64×64
        │                                    │
        ├──► Conv(64,s2) → BN → LeakyReLU   ├──► Concat
        │    Conv(128) → BN → LeakyReLU ───┐│    32×32
        │                                   ││
        ├──► Conv(128,s2) → BN → LeakyReLU ├┤ Skip
        │    Conv(256) → BN → LeakyReLU ───┐│    16×16
        │                                   ││
        ├──► Conv(256,s2) → BN → LeakyReLU ├┤ Connections
        │    Conv(256) → BN → LeakyReLU ───┐│    8×8
        │                                   │
        └───────────────────────────────────┘
                            │
                    ┌───────▼────────┐
                    │   BOTTLENECK   │
                    │ Conv(512,s1)   │
                    │ (8×8×512)      │
                    └───────┬────────┘
                            │
        ┌───────────────────┴───────────────────┐
        │            DECODER PATH               │
        │
        ├──► UpSample(2×) → Conv(256)           │    16×16
        │    Concat Skip ───────────┘
        │
        ├──► UpSample(2×) → Conv(128)           │    32×32
        │    Concat Skip ───────────┘
        │
        ├──► UpSample(2×) → Conv(64)            │    64×64
        │    Concat Skip ───────────┘
        │
        └───────────────────┬────────────────────┘
                            │
                    ┌───────▼────────┐
                    │ Conv(3,1,1)    │
                    │ Tanh           │
                    │ (64×64×3)      │
                    └───────┬────────┘
                            │
                    ┌───────▼────────┐
                    │ Perturbation   │
                    │ × α (0.05)     │
                    │ Clip [-1, 1]   │
                    └───────┬────────┘
                            │
                            ▼
                    ┌───────────────────┐
                    │  Signed Image     │
                    │  (64×64×3)        │
                    │  = Original +     │
                    │    α*Perturbation │
                    └───────────────────┘
```

## 4. Bob Network Architecture (CNN with Dual Heads)

```
                    BOB (ENCODER + DUAL HEADS)

Input: Image (64×64×3)

    ┌─────────────────────────────────────┐
    │    SHARED FEATURE EXTRACTION        │
    ├─────────────────────────────────────┤
    │                                     │
    │  Conv(32,s1) → BN → ReLU ────┐    │  64×64
    │  Conv(32,s2) → BN → ReLU ────┤    │  32×32
    │                               │    │
    │  Conv(64,s1) → BN → ReLU ────┼┐   │  32×32
    │  Conv(64,s2) → BN → ReLU ────┼┤   │  16×16
    │                               ││   │
    │  Conv(128,s1) → BN → ReLU ───┐├┤  │  16×16
    │  Conv(128,s2) → BN → ReLU ───┤├┤  │  8×8
    │                               │││  │
    │  Conv(256,s1) → BN → ReLU ────┐├┤ │  8×8
    │  Conv(256,s2) → BN → ReLU ────┼┤│ │  4×4
    │                               ├┼┤ │
    └───────────────────────────────┼┼┼─┘
                                    ││└──► 4×4×256
                    ┌───────────────┘│
                    │       Flatten  │
                    ▼                │
            ┌───────────────┐        │
            │ Dense(1024)   │        │
            │ + ReLU        │        │
            └───────┬───────┘        │
                    │                │
        ┌───────────┴────────────┬───┘
        │                        │
        ▼                        ▼
┌──────────────────┐    ┌──────────────────┐
│  HEAD 1: MESSAGE │    │  HEAD 2: AUTH    │
├──────────────────┤    ├──────────────────┤
│ Dense(512)       │    │ Dense(256)       │
│ + ReLU           │    │ + ReLU           │
│                  │    │ + Dropout(0.5)   │
│ Dense(256)       │    │                  │
│ + ReLU           │    │ Dense(128)       │
│                  │    │ + ReLU           │
│ Dense(msg_len)   │    │ + Dropout(0.3)   │
│ + Tanh           │    │                  │
│                  │    │ Dense(1)         │
│ Output: bits     │    │ + Sigmoid        │
│ [-1, 1] shape    │    │                  │
│ (batch, 256)     │    │ Output: prob     │
│                  │    │ [0, 1] shape     │
└──────────────────┘    │ (batch, 1)       │
                        └──────────────────┘
        │                       │
        │ extracted_bits        │ authenticity_prob
        └───────────┬───────────┘
                    ▼
        ┌───────────────────────┐
        │  DUAL OUTPUTS         │
        │  (bits, auth_score)   │
        └───────────────────────┘
```

## 5. Eve Network Architecture

```
                         EVE (SAME AS ALICE)

Same U-Net structure as Alice but with SEPARATE WEIGHTS

Purpose: Generate FORGED signatures that fool Bob

Input: Image (64×64×3) + Message (64×64×1)
       (Eve doesn't have the AES key - tries to forge)

Output: Forged Image (64×64×3)

Training Objective:
  - Fool Bob into accepting forged images
  - (Optionally) extract message bits
```

## 6. Training Loop Flow

```
                    ADVERSARIAL TRAINING LOOP

┌────────────────────────────────────────────────────┐
│         FOR EACH EPOCH (50 total)                  │
└────────────────────────────────────────────────────┘

  ┌──────────────────────────────────┐
  │  PHASE 1: ALICE + BOB TRAINING   │  (20 iterations)
  ├──────────────────────────────────┤
  │ • Alice embeds message            │
  │ • Bob extracts + classifies       │
  │ • Combined loss optimization      │
  │ Output: Better embedding/extraction
  └──────────────────────────────────┘
              │
              ▼
  ┌──────────────────────────────────┐
  │  PHASE 2: BOB CLASSIFIER         │  (20 iterations)
  ├──────────────────────────────────┤
  │ • Mixed batch: signed + unsigned │
  │ • Focus on authenticity           │
  │ Output: Better classification    │
  └──────────────────────────────────┘
              │
              ▼
  ┌──────────────────────────────────┐
  │  PHASE 3: EVE ATTACKS            │  (40 iterations)
  ├──────────────────────────────────┤
  │ • Eve creates forgeries           │
  │ • Bob frozen (not trained)        │
  │ Output: Strong adversary         │
  └──────────────────────────────────┘
              │
              ▼
  ┌──────────────────────────────────┐
  │  PHASE 4: BOB HARDENING          │  (10 iterations)
  ├──────────────────────────────────┤
  │ • Distinguish Alice vs Eve        │
  │ • Mixed: authentic + forged       │
  │ Output: Robust classifier        │
  └──────────────────────────────────┘
              │
              ▼
      ┌───────────────┐
      │  NEXT EPOCH   │
      └───────────────┘
```

## 7. Loss Function Computation

```
                    COMBINED LOSS FUNCTION

Training Phase (Alice + Bob):

    Loss = λ_recon × L_recon
         + λ_msg × L_msg
         + λ_auth × L_auth
         + λ_imper × L_imper

Where:

  L_recon = MSE(original, perturbed)
            ↳ Keep image close to original

  L_msg = MSE(original_bits, extracted_bits)
          ↳ Bob extracts correct message

  L_auth = BCE(auth_pred, ones)
           ↳ Bob classifies as authentic

  L_imper = max(0, ||perturbation||_∞ - ε)
            ↳ Bound perturbation magnitude

Weights (default):
  λ_recon = 1.0  ← Imperceptibility
  λ_msg = 2.0    ← Message extraction (most important)
  λ_auth = 1.0   ← Authentication
  λ_imper = 0.5  ← Perturbation bound


Eve's Loss:

  L_eve = L_fool_bob + 0.5 × L_extract
          ↳ Fool Bob (primary)
          ↳ Extract message (secondary)
```

## 8. Data Flow Diagram

```
              INPUT DATA PIPELINE

┌─────────────────┐
│   Raw Image     │  Any size, any format
│   [0, 1] or     │
│   [0, 255]      │
└────────┬────────┘
         │
         ▼
    ┌─────────────────────┐
    │ Preprocess          │  • Resize to 64×64
    │                     │  • Normalize to [-1, 1]
    └────────┬────────────┘
             │
             ▼
    ┌─────────────────────────┐
    │ Prepare Batch           │  Stack: (batch_size, 64, 64, 3)
    └────────┬────────────────┘
             │
             ▼
    ┌─────────────────────────┐
    │ AES Encrypt Message     │  "SECRET" → 256 bits in [-1, 1]
    └────────┬────────────────┘
             │
             ▼
    ┌─────────────────────────┐
    │ Alice Network           │  Embed watermark
    │ Input: (image, bits)    │
    └────────┬────────────────┘
             │
             ▼
    ┌─────────────────────────┐
    │ Signed Image            │  [-1, 1] imperceptible
    │ (64×64×3)               │
    └────────┬────────────────┘
             │
             ▼
    ┌─────────────────────────┐
    │ Postprocess             │  Normalize to [0, 255]
    └────────┬────────────────┘
             │
             ▼
    ┌─────────────────────────┐
    │ Output Image            │  [0, 255] uint8 for saving
    │ (64×64×3)               │
    └─────────────────────────┘


              OUTPUT (VERIFICATION) PIPELINE

┌─────────────────────┐
│ Received Image      │  Unknown origin
│ [0, 255]            │  (possibly tampered)
└────────┬────────────┘
         │
         ▼
    ┌─────────────────┐
    │ Preprocess      │  Normalize to [-1, 1]
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Bob Network     │  Extract and classify
    │ Output:         │
    │  • bits ∈ [-1,1]│
    │  • auth ∈ [0,1] │
    └────────┬────────┘
             │
      ┌──────┴──────┐
      │             │
      ▼             ▼
  ┌────────┐  ┌──────────────┐
  │ Bits   │  │ Auth Score   │
  └───┬────┘  └──────┬───────┘
      │               │
      ▼               ▼
  ┌─────────────┐  ┌──────────┐
  │ AES Decrypt │  │Threshold │
  │  Message    │  │> 0.5?    │
  └──────┬──────┘  └────┬─────┘
         │              │
         ▼              ▼
  ┌────────────┐  ┌──────────┐
  │ "SECRET"   │  │ Authentic
  │ (or None)  │  │ (Bool)
  └────────────┘  └──────────┘
         │              │
         └──────┬───────┘
                ▼
       ┌──────────────────┐
       │ VERIFICATION     │
       │ RESULT           │
       │ {                │
       │ is_authentic:    │
       │ confidence:      │
       │ message:         │
       │ bit_error_rate:  │
       │ }                │
       └──────────────────┘
```

## 9. Module Dependency Graph

```
                 NEURAL_IMAGE_AUTH
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
    CONFIG          CRYPTO             DATA
    ├────────┐      ├─────────┐        ├──────────┐
    │         │      │         │        │          │
    │         │      ▼         ▼        ▼          ▼
    │         │   AES_CIPHER  KEY_    DATAGEN  PREPROCESSING
    │         │                MANAGER
    │         │
    ▼         ▼
    MODELS ◄──┘
    ├────────┬────────┐
    │        │        │
    ▼        ▼        ▼
   ALICE   BOB      EVE
    │        │        │
    └────┬───┴────┬───┘
         │        │
         ▼        ▼
    TRAINING   INFERENCE
    ├────┬────┬────┐
    │    │    │    │
    ▼    ▼    ▼    ▼
  LOSSES  TRAINER  METRICS  ◄─ UTILS (save/load, etc)
    │        │
    └────┬───┘
         │
         ▼
      MAIN
```

---

This completes the complete Neural Image Authentication System with AES Encryption!
