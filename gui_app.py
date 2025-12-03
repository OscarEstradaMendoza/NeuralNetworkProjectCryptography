"""
GUI Application for Neural Image Authentication System

A user-friendly interface for signing and verifying images with AES-encrypted watermarks.
"""

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, scrolledtext
except ImportError as e:
    print("=" * 60)
    print("ERROR: tkinter is not available on this system.")
    print("=" * 60)
    print("\nThe desktop GUI (gui_app.py) requires tkinter, which is not")
    print("installed or configured in your Python environment.")
    print("\nSOLUTIONS:")
    print("1. Use the Web GUI instead (recommended):")
    print("   python web_gui.py")
    print("\n2. Install tkinter for your Python version:")
    print("   macOS: brew install python-tk")
    print("   Linux: sudo apt-get install python3-tk")
    print("   (Then reinstall Python or use system Python)")
    print("\n3. Use a different Python installation that includes tkinter")
    print("=" * 60)
    sys.exit(1)
import numpy as np
from PIL import Image, ImageTk
import os
import sys
import tensorflow as tf

# Configure device for Apple Silicon (M1 Pro)
try:
    from neural_image_auth.device_setup import configure_device
    configure_device()
except ImportError:
    # Fallback if device_setup not available
    tf.get_logger().setLevel("ERROR")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from neural_image_auth.models.alice import create_alice_network
from neural_image_auth.models.bob import create_bob_network
from neural_image_auth.models.eve import create_eve_network
from neural_image_auth.inference import NeuralImageAuthenticator
from neural_image_auth.crypto.key_manager import KeyManager
from neural_image_auth.utils import load_model, save_model
from neural_image_auth.config import KEY_DIR, LOG_DIR, IMAGE_SIZE
from neural_image_auth.data.preprocessing import preprocess_for_network, postprocess_from_network


class ImageAuthGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Image Authentication System")
        self.root.geometry("1000x700")

        # Initialize variables
        self.current_image = None
        self.current_image_path = None
        self.signed_image = None
        self.alice_model = None
        self.bob_model = None
        self.authenticator = None
        self.key_manager = KeyManager(KEY_DIR)

        # Create GUI widgets first so status_text is available for logging
        self.create_widgets()

        # Check if models exist, if not create them
        self.setup_models()

        # Create GUI
        # (layout already configured in create_widgets)
        
    def setup_models(self):
        """Load or create models."""
        try:
            # Try to find trained models
            model_paths = []
            if os.path.exists(LOG_DIR):
                for item in os.listdir(LOG_DIR):
                    train_dir = os.path.join(LOG_DIR, item)
                    if os.path.isdir(train_dir):
                        alice_path = os.path.join(train_dir, "models", "alice.keras")
                        bob_path = os.path.join(train_dir, "models", "bob.keras")
                        if os.path.exists(alice_path) and os.path.exists(bob_path):
                            model_paths.append((alice_path, bob_path))
            
            if model_paths:
                # Use most recent model
                latest = sorted(model_paths, key=lambda x: os.path.getmtime(x[0]), reverse=True)[0]
                self.status_text.insert(tk.END, f"Loading models from: {latest[0]}\n")
                self.root.update()
                self.alice_model = load_model(latest[0])
                self.bob_model = load_model(latest[1])
                self.status_text.insert(tk.END, "✓ Models loaded successfully!\n")
            else:
                # Create new models (untrained)
                self.status_text.insert(tk.END, "No trained models found. Creating new models...\n")
                self.root.update()
                self.alice_model = create_alice_network()
                self.bob_model = create_bob_network()
                self.status_text.insert(tk.END, "✓ New models created (untrained - results may vary)\n")
                self.status_text.insert(tk.END, "⚠ For best results, train models first using: python -m neural_image_auth.main\n")
            
            # Load or create AES key
            if self.key_manager.key_exists("default_key"):
                aes_key = self.key_manager.load_key("default_key")
            else:
                aes_key = self.key_manager.generate_key(key_size=16)
                self.key_manager.save_key(aes_key, "default_key")
            
            self.authenticator = NeuralImageAuthenticator(
                self.alice_model, 
                self.bob_model, 
                aes_key=aes_key
            )
            self.status_text.insert(tk.END, "✓ Authenticator initialized!\n")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to setup models: {str(e)}")
            self.status_text.insert(tk.END, f"✗ Error: {str(e)}\n")
    
    def create_widgets(self):
        """Create GUI widgets."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Left panel - Controls
        left_panel = ttk.Frame(main_frame)
        left_panel.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Image upload section
        upload_frame = ttk.LabelFrame(left_panel, text="Image Upload", padding="10")
        upload_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(upload_frame, text="Upload Image", command=self.upload_image).pack(fill=tk.X)
        self.image_path_label = ttk.Label(upload_frame, text="No image selected", wraplength=200)
        self.image_path_label.pack(pady=5)
        
        # Message input section
        message_frame = ttk.LabelFrame(left_panel, text="Message", padding="10")
        message_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(message_frame, text="Enter message to embed:").pack(anchor=tk.W)
        self.message_entry = ttk.Entry(message_frame, width=30)
        self.message_entry.insert(0, "AUTHENTIC")
        self.message_entry.pack(fill=tk.X, pady=5)
        
        # Action buttons
        action_frame = ttk.LabelFrame(left_panel, text="Actions", padding="10")
        action_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(action_frame, text="Sign Image", command=self.sign_image).pack(fill=tk.X, pady=2)
        
        # Download button (initially disabled)
        self.download_btn = ttk.Button(action_frame, text="📥 Download Signed Image", 
                                       command=self.download_signed_image, state=tk.DISABLED)
        self.download_btn.pack(fill=tk.X, pady=2)
        
        ttk.Button(action_frame, text="Verify Image", command=self.verify_image).pack(fill=tk.X, pady=2)
        
        # Results section
        results_frame = ttk.LabelFrame(left_panel, text="Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, height=10, width=40)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Right panel - Image display
        right_panel = ttk.Frame(main_frame)
        right_panel.grid(row=0, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Image display
        image_frame = ttk.LabelFrame(right_panel, text="Image Preview", padding="10")
        image_frame.pack(fill=tk.BOTH, expand=True)
        
        self.image_label = ttk.Label(image_frame, text="No image loaded", anchor=tk.CENTER)
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        ttk.Label(status_frame, text="Status:").pack(side=tk.LEFT)
        self.status_text = scrolledtext.ScrolledText(status_frame, height=4, width=100)
        self.status_text.pack(fill=tk.X, expand=True)
        self.status_text.insert(tk.END, "Ready. Please upload an image to begin.\n")
    
    def upload_image(self):
        """Handle image upload."""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Load and display image
                img = Image.open(file_path)
                self.current_image_path = file_path
                
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize for display (max 400x400)
                display_img = img.copy()
                display_img.thumbnail((400, 400), Image.Resampling.LANCZOS)
                
                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(display_img)
                self.image_label.config(image=photo, text="")
                self.image_label.image = photo  # Keep a reference
                
                # Store original image as numpy array
                img_array = np.array(img)
                if img_array.max() > 1.0:
                    img_array = img_array / 255.0  # Normalize to [0, 1]
                self.current_image = img_array
                
                self.image_path_label.config(text=os.path.basename(file_path))
                self.status_text.insert(tk.END, f"✓ Image loaded: {os.path.basename(file_path)}\n")
                self.status_text.see(tk.END)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
                self.status_text.insert(tk.END, f"✗ Error loading image: {str(e)}\n")
    
    def sign_image(self):
        """Sign the current image with the message."""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please upload an image first.")
            return
        
        if not self.authenticator:
            messagebox.showerror("Error", "Authenticator not initialized. Please check models.")
            return
        
        try:
            message = self.message_entry.get().strip()
            if not message:
                messagebox.showwarning("Warning", "Please enter a message.")
                return
            
            self.status_text.insert(tk.END, f"Signing image with message: '{message}'...\n")
            self.status_text.see(tk.END)
            self.root.update()
            
            # Sign the image
            signed = self.authenticator.sign_image(self.current_image, message)
            self.signed_image = signed
            
            # Display signed image
            signed_img = Image.fromarray(signed.astype(np.uint8))
            display_img = signed_img.copy()
            display_img.thumbnail((400, 400), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(display_img)
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo
            
            # Enable download button
            self.download_btn.config(state=tk.NORMAL)
            
            self.status_text.insert(tk.END, "✓ Image signed successfully!\n")
            self.status_text.insert(tk.END, "  Click 'Download Signed Image' to save it.\n")
            self.status_text.see(tk.END)
            
            # Update results
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Image signed with message: '{message}'\n")
            self.results_text.insert(tk.END, f"Signed image shape: {signed.shape}\n")
            self.results_text.insert(tk.END, "You can now verify or save the signed image.\n")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to sign image: {str(e)}")
            self.status_text.insert(tk.END, f"✗ Error signing image: {str(e)}\n")
    
    def download_signed_image(self):
        """Download the signed image to a file."""
        if self.signed_image is None:
            messagebox.showwarning("Warning", "No signed image available. Please sign an image first.")
            return
        
        try:
            # Get save file path
            if self.current_image_path:
                # Suggest filename based on original
                base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
                default_name = f"{base_name}_signed.png"
            else:
                default_name = "signed_image.png"
            
            file_path = filedialog.asksaveasfilename(
                title="Save Signed Image",
                defaultextension=".png",
                initialfile=default_name,
                filetypes=[
                    ("PNG files", "*.png"),
                    ("JPEG files", "*.jpg *.jpeg"),
                    ("All files", "*.*")
                ]
            )
            
            if file_path:
                # Convert numpy array to PIL Image and save
                img = Image.fromarray(self.signed_image.astype(np.uint8))
                img.save(file_path)
                
                self.status_text.insert(tk.END, f"✓ Signed image saved to: {file_path}\n")
                self.status_text.insert(tk.END, f"  Next step: Upload this file to verify it.\n")
                self.status_text.see(tk.END)
                
                messagebox.showinfo("Success", f"Signed image saved successfully!\n\nFile: {file_path}\n\nYou can now upload this image to verify its authenticity.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save image: {str(e)}")
            self.status_text.insert(tk.END, f"✗ Error saving image: {str(e)}\n")
    
    def verify_image(self):
        """Verify the current image."""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please upload an image first.")
            return
        
        if not self.authenticator:
            messagebox.showerror("Error", "Authenticator not initialized. Please check models.")
            return
        
        try:
            self.status_text.insert(tk.END, "Verifying image...\n")
            self.status_text.see(tk.END)
            self.root.update()
            
            # Verify the image (use signed image if available, otherwise original)
            image_to_verify = self.signed_image if self.signed_image is not None else self.current_image
            result = self.authenticator.verify_image(image_to_verify)
            
            # Display results
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "=== VERIFICATION RESULTS ===\n\n")
            self.results_text.insert(tk.END, f"Authentic: {'✓ YES' if result['is_authentic'] else '✗ NO'}\n")
            self.results_text.insert(tk.END, f"Confidence: {result['confidence']:.2%}\n")
            self.results_text.insert(tk.END, f"Bit Error Rate: {result['bit_error_rate']:.2%}\n")
            
            if result['extracted_message']:
                self.results_text.insert(tk.END, f"Extracted Message: '{result['extracted_message']}'\n")
            else:
                self.results_text.insert(tk.END, "Extracted Message: (Decryption failed)\n")
            
            self.status_text.insert(tk.END, "✓ Verification complete!\n")
            self.status_text.see(tk.END)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to verify image: {str(e)}")
            self.status_text.insert(tk.END, f"✗ Error verifying image: {str(e)}\n")
    
    def save_signed_image(self):
        """Save the signed image to file."""
        if self.signed_image is None:
            messagebox.showwarning("Warning", "No signed image to save. Please sign an image first.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Signed Image",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                signed_img = Image.fromarray(self.signed_image.astype(np.uint8))
                signed_img.save(file_path)
                self.status_text.insert(tk.END, f"✓ Signed image saved to: {file_path}\n")
                self.status_text.see(tk.END)
                messagebox.showinfo("Success", f"Image saved successfully to:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {str(e)}")
                self.status_text.insert(tk.END, f"✗ Error saving image: {str(e)}\n")


def main():
    """Launch the GUI application."""
    root = tk.Tk()
    app = ImageAuthGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

