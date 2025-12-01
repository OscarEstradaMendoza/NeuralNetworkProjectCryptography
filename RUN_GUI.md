# How to Run the GUI Application

## Quick Start (3 Steps)

### 1. Activate Virtual Environment
```bash
cd "/Users/oscarestradamendoza/Desktop/Cryptography_Project"
source venv/bin/activate
```

### 2. Install Dependencies (if needed)
```bash
pip install -r requirements.txt
```

### 3. Run the Web GUI (in Browser)
```bash
python web_gui.py
```

**OR** use the quick start script:
```bash
./start_gui.sh
```

### 4. Run the Desktop GUI (Tkinter Window)
```bash
python gui_app.py
```

### 4. Open Browser
For the **Web GUI**, the server will automatically find an available port and display it. Open your web browser and navigate to:
```
http://localhost:[PORT]
```
(Replace [PORT] with the port number shown in the terminal, e.g., 5001, 5002, etc.)

**Note**: If port 5000 is in use (common on macOS due to AirPlay), the server will automatically use a different port.

---

## What You'll See

A beautiful web interface with two panels:

### Left Panel: Sign Image
- Upload an image
- Enter a message to embed
- Click "Sign Image"
- See the signed image with embedded watermark

### Right Panel: Verify Image
- Upload an image to verify
- Click "Verify Image"
- See results:
  - ✓/✗ Authentic status
  - Confidence percentage
  - Extracted message
  - Bit error rate

---

## Optional: Train Models First (Recommended)

For best results, train the models first (takes ~5-10 minutes with 2 epochs):

```bash
# In a separate terminal, activate venv and run:
source venv/bin/activate
python -m neural_image_auth.main
```

Then restart the GUI - it will automatically use the trained models!

---

## Features

✅ **Easy to Use**: Simple web interface, no installation needed  
✅ **Real-time Preview**: See images before and after signing  
✅ **Secure**: AES-128-CBC encryption for messages  
✅ **Fast**: Quick signing and verification  
✅ **Beautiful UI**: Modern, responsive design  

---

## Troubleshooting

### Port 5000 already in use
**Solution**: Change the port in `web_gui.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Use port 5001 instead
```

### Models not loading
**Solution**: Train models first:
```bash
python -m neural_image_auth.main
```

### Browser shows "Connection refused"
**Solution**: Make sure the server is running. Check the terminal for errors.

---

## Example Workflow

1. **Start the server**: `python web_gui.py`
2. **Open browser**: Go to `http://localhost:5000`
3. **Sign an image**:
   - Upload a photo
   - Enter message: "MY_SECRET"
   - Click "Sign Image"
   - See the signed image
4. **Verify the image**:
   - Upload the signed image (or save and re-upload)
   - Click "Verify Image"
   - See the extracted message and authenticity check

---

Enjoy! 🚀

