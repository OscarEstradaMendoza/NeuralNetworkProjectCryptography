"""
Device setup utilities for Apple Silicon (M1/M2) optimization.

This module configures TensorFlow to use Metal Performance Shaders (MPS)
on Apple Silicon chips for GPU acceleration.
"""

import os
import sys
import platform
import tensorflow as tf

# Import config values with fallback defaults
try:
    from .config import USE_MPS, MEMORY_GROWTH, USE_MIXED_PRECISION
except ImportError:
    # Fallback defaults if config not available
    USE_MPS = True
    MEMORY_GROWTH = True
    USE_MIXED_PRECISION = False


def configure_device():
    """
    Configure TensorFlow for optimal performance on Apple Silicon.
    
    Sets up:
    - MPS (Metal Performance Shaders) backend for GPU acceleration
    - Memory growth to prevent OOM errors
    - Thread configuration for better CPU utilization
    """
    # Suppress TensorFlow warnings
    tf.get_logger().setLevel("ERROR")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Check if running on Apple Silicon
    try:
        # Try platform.machine() first (works on all platforms)
        machine = platform.machine()
        is_apple_silicon = machine == 'arm64'
    except:
        # Fallback: check sys.platform
        is_apple_silicon = sys.platform == 'darwin' and platform.processor() == 'arm'
    
    if is_apple_silicon and USE_MPS:
        # Configure MPS backend for Apple Silicon
        try:
            # List available physical devices
            physical_devices = tf.config.list_physical_devices()
            
            # Check for MPS (Metal Performance Shaders)
            mps_devices = [d for d in physical_devices if 'GPU' in d.name or 'MPS' in str(d)]
            
            if mps_devices:
                print(f"✓ Detected Apple Silicon GPU: {len(mps_devices)} device(s)")
                
                # Configure memory growth for MPS devices
                if MEMORY_GROWTH:
                    try:
                        for device in mps_devices:
                            tf.config.experimental.set_memory_growth(device, True)
                        print("✓ Memory growth enabled for GPU")
                    except Exception as e:
                        print(f"⚠ Could not set memory growth: {e}")
                
                # Set MPS as default device
                # TensorFlow automatically uses MPS if available on Apple Silicon
                print("✓ Using Metal Performance Shaders (MPS) backend")
            else:
                print("⚠ MPS devices not found, using CPU")
                
        except Exception as e:
            print(f"⚠ Could not configure MPS: {e}")
            print("  Falling back to CPU")
    
    # Configure CPU threads for better performance
    # M1 Pro has 8 performance cores + 2 efficiency cores
    # Use 6-8 threads for optimal performance
    try:
        tf.config.threading.set_intra_op_parallelism_threads(6)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        print("✓ CPU thread configuration optimized for M1 Pro")
    except Exception as e:
        print(f"⚠ Could not configure CPU threads: {e}")
    
    # Configure mixed precision if enabled
    if USE_MIXED_PRECISION:
        try:
            # Note: MPS has limited mixed precision support
            # This may cause issues, so it's disabled by default in config
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("✓ Mixed precision enabled (may have limited MPS support)")
        except Exception as e:
            print(f"⚠ Could not enable mixed precision: {e}")
    
    # Print device summary
    print_device_summary()


def print_device_summary():
    """Print summary of available devices."""
    print("\n" + "=" * 60)
    print("DEVICE CONFIGURATION")
    print("=" * 60)
    
    try:
        physical_devices = tf.config.list_physical_devices()
        
        print(f"Available devices:")
        for device in physical_devices:
            print(f"  - {device.name} ({device.device_type})")
        
        # Check if MPS is available
        if any('GPU' in d.name or 'MPS' in str(d) for d in physical_devices):
            print("\n✓ GPU acceleration available (MPS)")
        else:
            print("\n⚠ Using CPU (no GPU acceleration)")
            
    except Exception as e:
        print(f"⚠ Could not list devices: {e}")
    
    print("=" * 60 + "\n")


def get_device_info() -> dict:
    """
    Get information about available devices.
    
    Returns:
        Dictionary with device information
    """
    info = {
        "devices": [],
        "has_gpu": False,
        "has_mps": False,
        "device_count": 0,
    }
    
    try:
        physical_devices = tf.config.list_physical_devices()
        info["device_count"] = len(physical_devices)
        
        for device in physical_devices:
            device_info = {
                "name": device.name,
                "type": device.device_type,
            }
            info["devices"].append(device_info)
            
            if device.device_type == "GPU":
                info["has_gpu"] = True
                try:
                    machine = platform.machine()
                    if "MPS" in str(device) or machine == 'arm64':
                        info["has_mps"] = True
                except:
                    pass
                    
    except Exception as e:
        info["error"] = str(e)
    
    return info

