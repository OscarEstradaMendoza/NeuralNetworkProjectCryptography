# Web GUI Application - Neural Image Authentication System

## Quick Start Guide

### Step 1: Activate Virtual Environment

```bash
cd "/Users/oscarestradamendoza/Desktop/Cryptography_Project"
source venv/bin/activate
```

### Step 2: Install Dependencies (if not already installed)

```bash
pip install -r requirements.txt
```

### Step 3a: Run the Web GUI Application (in Browser)

```bash
python web_gui.py
```

Then open the printed `http://localhost:[PORT]` URL in your browser.

### Step 3b: Run the Desktop GUI Application (Tkinter Window)

```bash
python gui_app.py
```

This opens a native desktop window for image signing and verification.

---

## How to Use the GUI

### 1. **Upload an Image**
   - Click "Upload Image" button
   - Select any image file (PNG, JPG, JPEG, BMP, GIF)
   - The image will be displayed in the preview area

### 2. **Enter a Message**
   - Type your message in the "Message" field (default: "AUTHENTIC")
   - This message will be AES-encrypted and embedded as a watermark

### 3. **Sign the Image**
   - Click "Sign Image" button
   - The system will embed the encrypted message into the image
   - The signed image will appear in the preview
   - Results will be shown in the Results panel

### 4. **Verify the Image**
   - Click "Verify Image" button
   - The system will check if the image is authentic
   - Results will show:
     - ✓/✗ Authentic status
     - Confidence score (0-100%)
     - Extracted message (if decryption succeeds)
     - Bit Error Rate

### 5. **Save Signed Image**
   - Click "Save Signed Image" button
   - Choose location and filename
   - The signed image will be saved

---

## Features

- **Easy Image Upload**: Drag-and-drop style file selection
- **Real-time Preview**: See your images before and after signing
- **Message Embedding**: Encrypt and embed any text message
- **Verification**: Check image authenticity and extract messages
- **Status Updates**: Real-time status messages in the status bar

---

## Important Notes

### Model Training (Optional but Recommended)

For best results, train the models first:

```bash
# Train models (2 epochs - takes ~5-10 minutes)
python -m neural_image_auth.main
```

**Without training**: The GUI will create untrained models automatically, but results may vary significantly.

**With training**: The GUI will automatically find and use the most recently trained models from the `logs/` directory.

### Image Requirements

- **Format**: PNG, JPG, JPEG, BMP, GIF
- **Size**: Any size (will be automatically resized to 64×64)
- **Color**: RGB images (grayscale will be converted)

---

## Troubleshooting

### Issue: "No module named 'flask'"
**Solution**: Install Flask:
```bash
pip install flask
```

### Issue: Models not loading
**Solution**: Train models first:
```bash
python -m neural_image_auth.main
```

### Issue: Poor verification results
**Solution**: 
- Make sure you've trained the models (2 epochs minimum)
- Use the same image that was signed for verification
- Check that the message matches what was embedded

---

## Technical Details

- **Image Processing**: Images are automatically resized to 64×64 pixels
- **Encryption**: AES-128-CBC encryption for message security
- **Watermarking**: Imperceptible neural network-based watermarking
- **Verification**: Dual-head network (message extraction + authentication)

---

## Example Workflow

1. **Upload** a photo of your choice
2. **Enter** message: "MY_SECRET_MESSAGE"
3. **Sign** the image (embeds the encrypted watermark)
4. **Save** the signed image
5. **Upload** the saved signed image
6. **Verify** to see the extracted message and authenticity check

---

Enjoy using the Neural Image Authentication System! 🚀

