# 🕵️ Debugging Features Added to main.py

## Overview
We've added comprehensive debugging to track down the "Cannot copy out of meta tensor" error.

## Debugging Functions Added:

### 1. `debug_gpu_memory(stage)`
- Shows GPU memory usage at different stages
- Tracks allocated, reserved, and free memory
- Helps identify memory pressure issues

### 2. `debug_model_tensors(model, stage)`
- Counts meta vs working tensors
- Shows device distribution across model parameters
- Identifies which specific tensors are on meta device
- Returns count of problematic tensors

### 3. `debug_inference_inputs(input_ids, image_tensor, stage)`
- Validates input tensors before inference
- Checks shapes, devices, and dtypes
- Helps catch device mismatches early

### 4. `debug_tensor_operation(tensor, operation_name)`
- Tracks individual tensor operations
- Shows tensor properties at each step
- Warns about meta device tensors immediately

## Debug Points Added:

### Model Loading:
- ✅ Before model loading (memory baseline)
- ✅ After each model type loads (Qwen/LLaVA)
- ✅ Final model validation
- ✅ Meta tensor count in final state

### VLM Inference:
- ✅ Input validation before inference
- ✅ Model state before inference 
- ✅ Step-by-step LLaVA processing
- ✅ Tensor operations at each step
- ✅ Error analysis with traceback

### Specific LLaVA Steps:
1. Conversation setup
2. Text tokenization  
3. Image processing
4. Input tensor preparation
5. Model generation (where error likely occurs)
6. Output decoding

## How to Use:

1. **Run your script normally** - all debug output will appear
2. **Look for these patterns:**
   - `⚠️ Meta Parameters: X` - If X > 0, that's the problem
   - `🚨 META TENSOR ERROR DETECTED!` - Confirms this is the issue
   - Memory usage spikes indicating OOM
   - Device mismatches in tensor operations

## Expected Debug Output:

```
🔍 === GPU MEMORY DEBUG - Before Model Loading ===
   💾 Total GPU Memory: 16.00GB
   💾 Allocated: 0.12GB (0.8%)
   💾 Reserved: 0.12GB (0.8%)
   💾 Free: 15.88GB (99.2%)

🕵️ === MODEL TENSOR DEBUG - After LLaVA Loading ===
   📊 Total Parameters: 7242
   📊 Total Buffers: 1234  
   📊 Device Distribution: {'cuda:0': 8476}
   ⚠️ Meta Parameters: 0
   ⚠️ Meta Buffers: 0
   ✅ Working Parameters: 7242
   ✅ Working Buffers: 1234
```

## When Error Occurs:

```
❌ VLM inference error: Cannot copy out of meta tensor; no data!
🚨 META TENSOR ERROR DETECTED!
🕵️ === MODEL TENSOR DEBUG - During Generation Error ===
   ⚠️ Meta Parameters: 156  ← THIS IS THE PROBLEM!
   🔍 First 5 Meta Parameters: ['model.layers.0.weight', 'model.layers.1.bias', ...]
```

## Next Steps After Running Debug:

1. **If meta tensors found**: We know it's a device mapping issue
2. **If memory usage high**: We know it's insufficient VRAM
3. **If specific tensors listed**: We can target the fix
4. **If error in specific step**: We know exactly where it fails

This debugging will give us concrete evidence of what's happening! 🎯
