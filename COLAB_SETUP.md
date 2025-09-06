# Google Colab Optimization Guide for OpenEMMA

## Overview
This setup adds **4-bit and 8-bit quantization** support to OpenEMMA for running efficiently on Google Colab's limited GPU memory (typically 15GB).

## Key Features Added

### 1. Quantization Support
- **4-bit quantization**: Maximum memory savings (~75% reduction)
- **8-bit quantization**: Balanced performance and memory (~50% reduction)
- **No quantization**: Full precision (default if not specified)

### 2. Colab-Specific Optimizations
- Automatic GPU memory management (95% allocation)
- Flash Attention enabled when available
- Memory cache clearing
- CPU offloading for extreme cases

### 3. Model Compatibility
- ✅ **LLaVA models**: Full quantization support via native `load_4bit`/`load_8bit`
- ✅ **Qwen models**: Full quantization support via `BitsAndBytesConfig`
- ✅ **All VLM models**: Automatic device mapping and memory optimization

## Usage Examples

### For LLaVA with 4-bit quantization (Recommended for Colab)
```bash
python main.py --model_path llava --quantize 4bit --colab-mode
```

### For Qwen with 8-bit quantization
```bash
python main.py --model_path qwen --quantize 8bit --colab-mode
```

### No quantization (full precision)
```bash
python main.py --model_path llava --quantize none --colab-mode
```

## Memory Usage Comparison

| Configuration | Estimated Memory | Colab Compatible |
|---------------|------------------|------------------|
| Full Precision | ~15-20GB | ❌ Likely OOM |
| 8-bit Quantization | ~7-10GB | ✅ Good |
| 4-bit Quantization | ~4-6GB | ✅ Excellent |

## Colab Setup Script

```python
# In your Colab notebook:
!git clone https://github.com/yasinshahid/OpenEMMA.git
%cd OpenEMMA

# Install dependencies
!pip install -r requirements.txt
!pip install bitsandbytes  # For quantization

# Test quantization setup
!python test_quantization.py

# Run with optimized settings
!python main.py --model_path llava --quantize 4bit --colab-mode --image_path "assets/scene-0061.jpg" --question "What do you see?"
```

## Troubleshooting

### If you get "Cannot copy out of meta tensor" error:
- This quantization setup should resolve the meta tensor issues
- The 4-bit quantization forces proper tensor materialization
- Memory pressure is reduced, preventing meta device fallbacks

### If you get OOM (Out of Memory) errors:
- Try 4-bit quantization: `--quantize 4bit`
- Enable Colab mode: `--colab-mode`
- Restart runtime and clear all variables before running

### Performance Considerations:
- **4-bit**: Fastest loading, smallest memory, slight quality loss
- **8-bit**: Balanced performance, good quality retention
- **Full precision**: Best quality, but may not fit in Colab memory

## Technical Details

The implementation adds:
1. `BitsAndBytesConfig` for Transformers models
2. `load_4bit`/`load_8bit` parameters for LLaVA
3. Automatic memory optimization
4. Flash Attention when available
5. Smart device mapping for Colab GPUs

This should resolve both the memory constraints and the "meta tensor" issues you were experiencing!
