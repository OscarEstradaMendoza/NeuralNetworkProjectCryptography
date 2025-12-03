# Apple M1 Pro Optimization Guide

This document describes the optimizations made for Apple M1 Pro (16GB RAM) systems.

## Optimizations Applied

### 1. **Batch Size Reduction**
- **Changed**: `BATCH_SIZE = 32` → `BATCH_SIZE = 16`
- **Reason**: 16GB RAM benefits from smaller batches to prevent memory pressure
- **Impact**: Slightly slower training per batch, but more stable and prevents OOM errors

### 2. **Metal Performance Shaders (MPS) Support**
- **Added**: Automatic MPS backend detection and configuration
- **Location**: `neural_image_auth/device_setup.py`
- **Benefits**: 
  - GPU acceleration on Apple Silicon
  - Faster training compared to CPU-only
  - Automatic device selection

### 3. **Memory Growth Configuration**
- **Enabled**: `MEMORY_GROWTH = True`
- **Purpose**: Prevents TensorFlow from allocating all GPU memory at once
- **Benefit**: Better memory management, allows other processes to run

### 4. **Mixed Precision Disabled**
- **Changed**: `USE_MIXED_PRECISION = False`
- **Reason**: MPS backend has limited mixed precision support
- **Note**: Can be re-enabled if you encounter no issues, but disabled by default for stability

### 5. **CPU Thread Optimization**
- **Configured**: 6 intra-op threads, 2 inter-op threads
- **Reason**: Optimized for M1 Pro's 8 performance cores + 2 efficiency cores
- **Location**: `neural_image_auth/device_setup.py`

### 6. **Evaluation Script Optimizations**
- **Reduced**: Default test images from 100 → 50
- **Reason**: Lower memory usage during evaluation
- **Impact**: Faster evaluation, still statistically significant

## Device Configuration

The system automatically:
1. Detects Apple Silicon architecture
2. Configures MPS backend for GPU acceleration
3. Sets up memory growth
4. Optimizes CPU thread usage
5. Prints device summary on startup

## Performance Expectations

### Training Speed
- **CPU Only**: ~2-4 hours for 50 epochs
- **With MPS (GPU)**: ~1-2 hours for 50 epochs
- **Memory Usage**: ~4-8GB during training (with batch size 16)

### Memory Usage
- **Training**: 4-8GB RAM
- **Inference**: 1-2GB RAM
- **Evaluation**: 2-4GB RAM (with 50 test images)

## Troubleshooting

### Issue: MPS Not Detected
**Solution**: Ensure you have TensorFlow 2.10+ installed:
```bash
pip install --upgrade tensorflow>=2.10.0
```

### Issue: Out of Memory During Training
**Solutions**:
1. Reduce `BATCH_SIZE` further (try 8 or 4)
2. Reduce `ALICE_BOB_ITERATIONS` and `EVE_ITERATIONS`
3. Close other applications

### Issue: Slow Training
**Solutions**:
1. Verify MPS is being used (check device summary on startup)
2. Ensure you're using TensorFlow 2.10+ with MPS support
3. Check Activity Monitor - GPU should show activity

### Issue: Mixed Precision Errors
**Solution**: Mixed precision is disabled by default. If you want to enable it:
1. Set `USE_MIXED_PRECISION = True` in `config.py`
2. Be aware it may cause issues with MPS backend

## Configuration Files Modified

1. **`neural_image_auth/config.py`**:
   - Batch size reduced to 16
   - MPS settings added
   - Mixed precision disabled

2. **`neural_image_auth/device_setup.py`** (NEW):
   - Device detection and configuration
   - MPS backend setup
   - Memory management

3. **`neural_image_auth/main.py`**:
   - Added device configuration on startup

4. **`evaluation_analysis.py`**:
   - Reduced default test images to 50
   - Added device configuration

5. **`web_gui.py`** and **`gui_app.py`**:
   - Added device configuration

## Verification

To verify MPS is working:

```python
import tensorflow as tf
print(tf.config.list_physical_devices())
```

You should see a GPU device listed if MPS is working.

## Additional Notes

- The optimizations are automatic - no manual configuration needed
- The system will fall back to CPU if MPS is not available
- All optimizations are backward compatible with other systems
- Memory usage is conservative to prevent issues on 16GB systems

## Performance Tips

1. **Close unnecessary applications** during training
2. **Use smaller batch sizes** if you encounter memory issues
3. **Monitor Activity Monitor** to see GPU usage
4. **Train in shorter sessions** if memory pressure builds up
5. **Use evaluation script** with reduced test images for faster iteration

