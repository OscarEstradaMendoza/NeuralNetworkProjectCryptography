# Evaluation Analysis Guide

This guide explains how to use the `evaluation_analysis.py` script to analyze the performance of the neural image authentication system.

## Overview

The evaluation script performs two types of analysis:

1. **Epoch Analysis**: Trains models with different numbers of epochs and evaluates performance metrics
2. **Message Length Analysis**: Tests performance with different message string lengths (characters)

## Metrics Evaluated

For each configuration, the script measures:

- **Authenticity Rate**: Fraction of images correctly identified as authentic
- **Average Confidence**: Mean confidence score from Bob's classifier
- **Average Bit Error Rate**: Average bit error rate in message extraction
- **Decryption Success Rate**: Fraction of messages successfully decrypted

## Usage

### Basic Usage

Run both analyses with default parameters:

```bash
python evaluation_analysis.py
```

### Epoch Analysis Only

Train models with different epoch counts (default: 1, 2, 5, 10):

```bash
python evaluation_analysis.py --epoch-analysis --epochs 1 2 5 10 20
```

### Message Length Analysis Only

Test with different message string lengths (default: 8, 16, 32, 64 characters):

```bash
python evaluation_analysis.py --message-length-analysis --message-lengths 8 16 32 64 128
```

### Custom Parameters

```bash
python evaluation_analysis.py \
    --epoch-analysis \
    --epochs 1 2 5 10 \
    --num-test-images 200 \
    --save-dir my_evaluation_results
```

## Command-Line Arguments

- `--epoch-analysis`: Run epoch analysis
- `--message-length-analysis`: Run message length analysis
- `--epochs EPOCHS`: List of epoch values to test (default: [1, 2, 5, 10])
- `--message-lengths LENGTHS`: List of message string lengths to test (default: [8, 16, 32, 64])
- `--num-test-images N`: Number of test images to use (default: 100)
- `--num-epochs-for-message-length N`: Epochs to train for message length analysis (default: 5)
- `--save-dir DIR`: Directory to save results (default: evaluation_results)

## Output

The script generates:

1. **JSON Results Files**:
   - `epoch_analysis_results.json`: Results for epoch analysis
   - `message_length_analysis_results.json`: Results for message length analysis
   - `all_evaluation_results.json`: Combined results

2. **Visualization Plots**:
   - `epoch_analysis.png`: 4-panel plot showing metrics vs epochs
   - `message_length_analysis.png`: 4-panel plot showing metrics vs message length

## Example Output

### Epoch Analysis Results

```
Results for 5 epochs:
  Authenticity Rate:     95.000%
  Avg Confidence:         0.923
  Avg Bit Error Rate:     2.500%
  Decryption Success:     92.000%
```

### Message Length Analysis Results

```
Results for message string length 32 (256 bits after encryption):
  Authenticity Rate:     94.000%
  Avg Confidence:         0.915
  Avg Bit Error Rate:     3.200%
  Decryption Success:     90.000%
```

## Notes

- **Epoch Analysis**: Each epoch configuration requires full training, which can be time-consuming
- **Message Length Analysis**: Trains one model and tests with different message sizes (more efficient)
- **Test Dataset**: Uses a fixed test dataset for consistent evaluation across configurations
- **Memory**: Ensure sufficient memory for loading multiple models during evaluation

## Troubleshooting

### Out of Memory

If you run out of memory, reduce `--num-test-images` or run analyses separately.

### Training Errors

If training fails for certain configurations, the script will continue with other configurations and report errors.

### Model Loading Issues

If you see model loading errors, ensure models were trained with the current codebase version (custom layers must match).

## Performance Tips

1. **Parallel Training**: Run epoch analysis configurations in parallel on different machines/GPUs
2. **Caching**: Results are saved to JSON, so you can re-plot without re-running evaluation
3. **Reduced Testing**: Use fewer test images (`--num-test-images 50`) for faster evaluation during development

