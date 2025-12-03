"""
Web-based GUI Application for Neural Image Authentication System

A Flask-based web interface for signing and verifying images with AES-encrypted watermarks.
Run with: python web_gui.py
Then open: http://localhost:5000
"""

# ============================================================================
# IMPORTS
# ============================================================================

# Flask web framework components
from flask import Flask, render_template, request, jsonify, send_file

# Image processing and data handling
import numpy as np  # Numerical operations for image arrays
from PIL import Image  # Image manipulation (open, convert, save)
import io  # In-memory file operations (BytesIO)
import os  # File system operations
import base64  # Encoding images for web display

# Deep learning framework
import tensorflow as tf

# Configure device for Apple Silicon (M1 Pro)
try:
    from neural_image_auth.device_setup import configure_device
    configure_device()
except ImportError:
    # Fallback if device_setup not available
    tf.get_logger().setLevel("ERROR")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import neural network models (Alice = encoder, Bob = decoder/classifier)
from neural_image_auth.models.alice import create_alice_network
from neural_image_auth.models.bob import create_bob_network

# Import authentication system components
from neural_image_auth.inference import NeuralImageAuthenticator  # Main API for signing/verifying
from neural_image_auth.crypto.key_manager import KeyManager  # AES key management
from neural_image_auth.utils import load_model  # Load saved trained models
from neural_image_auth.config import KEY_DIR, LOG_DIR, IMAGE_SIZE  # Configuration constants

# ============================================================================
# FLASK APP INITIALIZATION
# ============================================================================

# Create Flask application instance
app = Flask(__name__)

# Configure maximum file upload size (16MB) to prevent large file uploads
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# ============================================================================
# GLOBAL STATE VARIABLES
# ============================================================================

# Global authenticator instance (initialized at startup)
# This holds the trained Alice and Bob models along with the AES encryption key
authenticator = None

# Key manager for handling AES encryption keys (loads/generates keys from disk)
key_manager = KeyManager(KEY_DIR)


def setup_models():
    """
    Initialize the neural network models and authentication system.
    
    This function:
    1. Searches for previously trained models in the logs directory
    2. Loads the most recent trained models if found
    3. Creates new (untrained) models if none are found
    4. Loads or generates an AES encryption key
    5. Initializes the NeuralImageAuthenticator with models and key
    
    The authenticator is stored in the global variable for use by API endpoints.
    """
    global authenticator
    
    try:
        alice_model = None
        bob_model = None
        
        # Step 1: Try to find trained models in logs directory
        # Look for training runs in format: logs/train_YYYYMMDD_HHMMSS/
        if os.path.exists(LOG_DIR):
            # Sort directories by name (reverse = newest first)
            for item in sorted(os.listdir(LOG_DIR), reverse=True):
                train_dir = os.path.join(LOG_DIR, item)
                if os.path.isdir(train_dir):
                    # Check if both Alice and Bob models exist in this training run
                    alice_path = os.path.join(train_dir, "models", "alice.keras")
                    bob_path = os.path.join(train_dir, "models", "bob.keras")
                    if os.path.exists(alice_path) and os.path.exists(bob_path):
                        # Found trained models - load them
                        print(f"Loading models from: {alice_path}")
                        alice_model = load_model(alice_path)
                        bob_model = load_model(bob_path)
                        print("✓ Models loaded successfully!")
                        break  # Use the first (most recent) trained models found
        
        # Step 2: Create new models if no trained models were found
        if alice_model is None or bob_model is None:
            print("No trained models found. Creating new models...")
            # Create untrained models (will have random weights)
            alice_model = create_alice_network()  # Encoder network
            bob_model = create_bob_network()  # Decoder/classifier network
            print("✓ New models created (untrained - results may vary)")
            print("⚠ For best results, train models first using: python -m neural_image_auth.main")
        
        # Step 3: Load or create AES encryption key
        # The key is used to encrypt messages before embedding them in images
        if key_manager.key_exists("default_key"):
            # Key already exists - load it
            aes_key = key_manager.load_key("default_key")
        else:
            # No key found - generate a new 16-byte (128-bit) AES key
            aes_key = key_manager.generate_key(key_size=16)
            key_manager.save_key(aes_key, "default_key")
        
        # Step 4: Initialize the authenticator with models and encryption key
        # This combines Alice (encoder), Bob (decoder), and AES encryption
        authenticator = NeuralImageAuthenticator(
            alice_model,  # Model that embeds watermarks into images
            bob_model,    # Model that extracts watermarks and verifies authenticity
            aes_key=aes_key  # AES-128 key for message encryption
        )
        print("✓ Authenticator initialized!")
        
    except Exception as e:
        print(f"✗ Error setting up models: {str(e)}")
        raise


def image_to_base64(image_array):
    """
    Convert a numpy image array to a base64-encoded data URL string.
    
    This allows images to be displayed directly in HTML without saving to disk.
    The browser can render the image using: <img src="data:image/png;base64,...">
    
    Args:
        image_array: NumPy array representing an image (values should be 0-255)
    
    Returns:
        String in format: "data:image/png;base64,<encoded_image_data>"
    """
    # Convert numpy array to PIL Image (uint8 = 0-255 pixel values)
    img = Image.fromarray(image_array.astype(np.uint8))
    
    # Create in-memory buffer to hold PNG image data
    buffered = io.BytesIO()
    
    # Save image as PNG to the in-memory buffer
    img.save(buffered, format="PNG")
    
    # Encode the PNG bytes to base64 string
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    # Return as data URL (browser can display this directly)
    return f"data:image/png;base64,{img_str}"


def process_image(image_file):
    """
    Process an uploaded image file for use with neural networks.
    
    This function:
    1. Reads the uploaded file into memory
    2. Converts to RGB color mode (handles grayscale, RGBA, etc.)
    3. Converts to numpy array
    4. Normalizes pixel values to [0, 1] range (required by neural networks)
    
    Args:
        image_file: Flask file upload object (from request.files)
    
    Returns:
        NumPy array of shape (height, width, 3) with values in [0, 1]
    """
    # Read uploaded file into memory and open as PIL Image
    img = Image.open(io.BytesIO(image_file.read()))
    
    # Convert to RGB color mode if needed (handles grayscale, RGBA, etc.)
    # Neural networks expect RGB (3 channels)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Convert PIL Image to NumPy array
    img_array = np.array(img)
    
    # Normalize pixel values from [0, 255] to [0, 1] if needed
    # Neural networks work better with normalized values
    if img_array.max() > 1.0:
        img_array = img_array / 255.0
    
    return img_array


# ============================================================================
# FLASK ROUTES (API ENDPOINTS)
# ============================================================================

@app.route('/')
def index():
    """
    Serve the main web page.
    
    Returns:
        Rendered HTML template (templates/index.html) containing the web UI
    """
    return render_template('index.html')


@app.route('/api/sign', methods=['POST'])
def sign_image():
    """
    API endpoint to sign an image with an encrypted watermark.
    
    This endpoint:
    1. Receives an image file and text message from the client
    2. Encrypts the message using AES encryption
    3. Embeds the encrypted message into the image using Alice network
    4. Returns the signed image as a base64-encoded data URL
    
    Request format:
        - POST with form-data containing:
          - 'image': image file (PNG, JPG, etc.)
          - 'message': text string to embed
    
    Returns:
        JSON response with:
        - success: boolean
        - signed_image: base64-encoded image data URL
        - message: confirmation message
        OR error message with HTTP status code
    """
    try:
        # Validate that image file was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        # Validate that message was provided
        if 'message' not in request.form:
            return jsonify({'error': 'No message provided'}), 400
        
        # Extract image file and message from request
        image_file = request.files['image']
        message = request.form['message'].strip()
        
        # Validate message is not empty
        if not message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        # Check that authenticator is initialized
        if not authenticator:
            return jsonify({'error': 'Authenticator not initialized'}), 500
        
        # Step 1: Process uploaded image (convert to numpy array, normalize)
        image_array = process_image(image_file)
        
        # Step 2: Sign the image using neural network
        # This encrypts the message and embeds it imperceptibly into the image
        signed = authenticator.sign_image(image_array, message)
        
        # Step 3: Convert signed image to base64 for web display
        signed_b64 = image_to_base64(signed)
        
        # Return success response with signed image
        return jsonify({
            'success': True,
            'signed_image': signed_b64,  # Base64 data URL for <img src="...">
            'message': f'Image signed with message: "{message}"'
        })
        
    except Exception as e:
        # Return error response if anything goes wrong
        return jsonify({'error': str(e)}), 500


@app.route('/api/download_signed', methods=['POST'])
def download_signed():
    """
    API endpoint to download the signed image.
    
    This endpoint:
    1. Receives a base64-encoded signed image
    2. Converts it back to binary PNG format
    3. Returns it as a downloadable file
    
    Request format:
        - POST with JSON containing:
          - 'signed_image': base64-encoded image data URL
          - 'filename': optional filename (default: 'signed_image.png')
    
    Returns:
        PNG file download
    """
    try:
        data = request.get_json()
        
        if not data or 'signed_image' not in data:
            return jsonify({'error': 'No signed image data provided'}), 400
        
        # Extract base64 image data (remove data URL prefix if present)
        signed_image_b64 = data['signed_image']
        if signed_image_b64.startswith('data:image'):
            signed_image_b64 = signed_image_b64.split(',')[1]
        
        # Decode base64 to binary
        image_binary = base64.b64decode(signed_image_b64)
        
        # Get filename (default to signed_image.png)
        filename = data.get('filename', 'signed_image.png')
        if not filename.endswith('.png'):
            filename += '.png'
        
        # Create file-like object and send as download
        return send_file(
            io.BytesIO(image_binary),
            mimetype='image/png',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/verify', methods=['POST'])
def verify_image():
    """
    API endpoint to verify an image's authenticity and extract embedded message.
    
    This endpoint:
    1. Receives an image file (potentially signed with a watermark)
    2. Uses Bob network to extract the embedded encrypted message
    3. Decrypts the message using AES
    4. Checks if the image is authentic (has valid watermark)
    5. Returns verification results including extracted message
    
    Request format:
        - POST with form-data containing:
          - 'image': image file to verify
    
    Returns:
        JSON response with:
        - success: boolean
        - is_authentic: whether image contains valid watermark
        - confidence: authenticity confidence score (0-100%)
        - extracted_message: decrypted message from image (if successful)
        - bit_error_rate: percentage of bits that didn't match (0-100%)
        - message: confirmation message
        OR error message with HTTP status code
    """
    try:
        # Validate that image file was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        # Check that authenticator is initialized
        if not authenticator:
            return jsonify({'error': 'Authenticator not initialized'}), 500
        
        # Step 1: Process uploaded image (convert to numpy array, normalize)
        image_file = request.files['image']
        image_array = process_image(image_file)
        
        # Step 2: Verify the image using neural network
        # This extracts the watermark, decrypts it, and checks authenticity
        result = authenticator.verify_image(image_array)
        
        # Step 3: Return verification results
        return jsonify({
            'success': True,
            'is_authentic': result['is_authentic'],  # Boolean: is image authentic?
            'confidence': round(result['confidence'] * 100, 2),  # 0-100% confidence
            'extracted_message': result['extracted_message'],  # Decrypted message (or empty if failed)
            'bit_error_rate': round(result['bit_error_rate'] * 100, 2),  # 0-100% error rate
            'message': 'Verification complete'
        })
        
    except Exception as e:
        # Return error response if anything goes wrong
        return jsonify({'error': str(e)}), 500


@app.route('/api/status', methods=['GET'])
def status():
    """
    API endpoint to check if the authentication system is ready.
    
    This is a health check endpoint that the frontend can call to verify
    that models are loaded and the system is operational.
    
    Returns:
        JSON response with:
        - authenticator_ready: boolean indicating if system is ready
        - message: human-readable status message
    """
    return jsonify({
        'authenticator_ready': authenticator is not None,  # True if models loaded
        'message': 'System ready' if authenticator else 'System not ready'
    })


# ============================================================================
# APPLICATION STARTUP
# ============================================================================

if __name__ == '__main__':
    """
    Main entry point when running the script directly.
    
    This section:
    1. Initializes the neural network models
    2. Finds an available port (handles port conflicts)
    3. Starts the Flask web server
    """
    
    # Print startup banner
    print("=" * 60)
    print("Neural Image Authentication System - Web GUI")
    print("=" * 60)
    print("\nInitializing models...")
    
    # Step 1: Load or create neural network models
    # This must complete before starting the web server
    setup_models()
    
    print("\n" + "=" * 60)
    print("Starting web server...")
    
    # Step 2: Find an available port dynamically
    # This avoids conflicts if port 5000 is already in use (common on macOS)
    import socket
    
    def find_free_port():
        """
        Find an available TCP port by binding to port 0 (OS assigns free port).
        
        Returns:
            Integer port number that is currently available
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))  # Bind to any available port
            return s.getsockname()[1]  # Return the assigned port number
    
    # Get an available port
    port = find_free_port()
    print(f"Using port: {port}")
    print(f"Open your browser and go to: http://localhost:{port}")
    print("=" * 60 + "\n")
    
    # Step 3: Start Flask development server
    try:
        app.run(
            debug=True,           # Enable debug mode (shows errors in browser)
            host='0.0.0.0',       # Listen on all network interfaces
            port=port,            # Use the dynamically found port
            use_reloader=False    # Disable auto-reload (prevents model reload issues)
        )
    except OSError as e:
        # Handle case where port becomes unavailable between check and binding
        if "Address already in use" in str(e):
            print(f"\n⚠ Port {port} is in use. Trying alternative port...")
            # Find a different port and try again
            port = find_free_port()
            print(f"Using port: {port}")
            print(f"Open your browser and go to: http://localhost:{port}\n")
            app.run(debug=True, host='0.0.0.0', port=port, use_reloader=False)
        else:
            # Re-raise other OSErrors (permission denied, etc.)
            raise

