# Neural Image Authentication Workflow Guide

## Complete 3-Step Workflow

This guide explains how to use the updated GUI applications to sign, download, and verify images with embedded encrypted watermarks.

---

## Prerequisites

1. **Ensure models are trained** (at least 20+ epochs recommended):
```bash
cd /Users/oscarestradamendoza/.cursor/worktrees/Cryptography_Project/nik
source venv/bin/activate
python -m neural_image_auth.main
```

2. **Activate virtual environment** before running GUIs:
```bash
source venv/bin/activate
```

---

## STEP 1: Sign an Image

### Web GUI (`web_gui.py`):
1. Run: `python web_gui.py`
2. Open browser: `http://localhost:5000`
3. In the **Sign Image** panel (left side):
   - Click "Choose File" and select an image
   - Enter a message to embed (default: "AUTHENTIC")
   - Click "Sign Image"
4. **Wait for signing to complete** (~1-5 seconds)
5. You'll see the signed image preview appear below

### Desktop GUI (`gui_app.py`):
1. Run: `python gui_app.py`
2. Click "Upload Image" and select an image
3. Enter a message in the "Message to Embed" field
4. Click "Sign Image"
5. The signed image will appear in the preview area

---

## STEP 2: Download the Watermarked Image

### Web GUI:
1. After signing, a green **"📥 Download Signed Image"** button appears above the preview
2. Click the button
3. Your browser will download the file (e.g., `your_image_signed.png`)
4. Save it to a known location

### Desktop GUI:
1. After signing, the **"📥 Download Signed Image"** button becomes enabled
2. Click the button
3. Choose where to save the file
4. Default filename: `[original_name]_signed.png`

**Important Notes:**
- The downloaded image contains the embedded watermark
- Do NOT use the original image for verification
- The signed image looks identical to the original (imperceptible watermark)

---

## STEP 3: Verify the Watermarked Image

### Web GUI:
1. In the **Verify Image** panel (right side):
   - Click "Choose File" 
   - Select the **downloaded signed image** (from Step 2)
   - Click "Verify Image"
2. Results will show:
   - ✓ **AUTHENTIC** or ✗ **NOT AUTHENTIC**
   - **Confidence**: percentage (should be ~90-100% for trained models)
   - **Bit Error Rate**: percentage (should be ~0-10% for good models)
   - **Extracted Message**: the decrypted message

### Desktop GUI:
1. Click "Upload Image" again
2. Select the **downloaded signed image** (from Step 2)
3. Click "Verify Image"
4. Results appear in the results panel showing:
   - Authentic: ✓ YES / ✗ NO
   - Confidence percentage
   - Bit Error Rate
   - Extracted Message

---

## Expected Results

### Well-Trained Models (20+ epochs):
- **Authenticity Rate**: ✓ AUTHENTIC (90-100%)
- **Confidence**: 90-99%
- **Bit Error Rate**: 0-10%
- **Decryption Success**: Message successfully extracted

### Poorly-Trained Models (< 10 epochs):
- **Authenticity Rate**: May show authentic, but confidence is low
- **Confidence**: 50-70%
- **Bit Error Rate**: 40-100%
- **Decryption Success**: Often fails (message shows as "(Decryption failed)")

---

## Troubleshooting

### Issue: "No trained models found"
**Solution**: Train models first:
```bash
python -m neural_image_auth.main
```

### Issue: Download button doesn't appear (Web GUI)
**Solution**: 
- Make sure signing completed successfully
- Check browser console for errors (F12)
- Refresh the page and try again

### Issue: Verification shows "NOT AUTHENTIC"
**Possible Causes**:
1. Using the original image instead of the signed/downloaded image
2. Image was modified after signing (compression, editing, etc.)
3. Using wrong AES key (different from training)
4. Models not properly trained

### Issue: Extracted message is "(Decryption failed)"
**Possible Causes**:
1. Models insufficiently trained (< 20 epochs)
2. High bit error rate (> 30%)
3. Image was modified after signing
4. AES key mismatch

### Issue: Confidence is very low (~50%)
**Solution**: Train models for more epochs (aim for 50+ epochs)

---

## Complete Example Workflow

### Using Web GUI:

```bash
# Terminal 1: Start web server
cd /Users/oscarestradamendoza/.cursor/worktrees/Cryptography_Project/nik
source venv/bin/activate
python web_gui.py
```

1. Open `http://localhost:5000` in browser
2. **Sign Section** (left):
   - Upload `test_image.jpg`
   - Message: "SECRET123"
   - Click "Sign Image"
   - Click "📥 Download Signed Image"
   - Save as `test_image_signed.png`
3. **Verify Section** (right):
   - Upload `test_image_signed.png` (the downloaded file)
   - Click "Verify Image"
   - Check results:
     - ✓ AUTHENTIC
     - Confidence: 95.23%
     - BER: 3.12%
     - Extracted Message: "SECRET123"

### Using Desktop GUI:

```bash
# Terminal: Start desktop app
cd /Users/oscarestradamendoza/.cursor/worktrees/Cryptography_Project/nik
source venv/bin/activate
python gui_app.py
```

1. Click "Upload Image" → select `photo.jpg`
2. Message: "CONFIDENTIAL"
3. Click "Sign Image"
4. Click "📥 Download Signed Image" → save as `photo_signed.png`
5. Click "Upload Image" again → select `photo_signed.png`
6. Click "Verify Image"
7. Check results in the Results panel

---

## Tips for Best Results

1. **Train for 50+ epochs** for production-quality results
2. **Use PNG format** for downloads (lossless, preserves watermark)
3. **Avoid JPEG** for signed images (lossy compression damages watermark)
4. **Test with different messages** to ensure robustness
5. **Save the AES key** (`keys/default_key`) - needed for decryption

---

## File Locations

- **Trained models**: `logs/train_YYYYMMDD_HHMMSS/models/`
- **AES keys**: `keys/default_key`
- **Downloaded signed images**: Browser downloads folder (Web GUI) or chosen location (Desktop GUI)
- **Training history**: `logs/train_YYYYMMDD_HHMMSS/training_history.png`

---

## Next Steps

1. **Train more epochs** if results are poor
2. **Adjust hyperparameters** in `neural_image_auth/config.py`
3. **Test robustness** by adding noise, compression, rotation
4. **Integrate into applications** using the Python API

For more information, see:
- `README_NEW.md` - Comprehensive documentation
- `QUICKSTART.md` - Quick start guide
- `M1_PRO_OPTIMIZATION.md` - Performance optimization for M1 Pro

