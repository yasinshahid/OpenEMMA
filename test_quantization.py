#!/usr/bin/env python3
"""
Test script for Colab quantization functionality
"""
import torch
from transformers import BitsAndBytesConfig

def get_colab_quantization_config(quantize_mode="4bit"):
    """
    Create quantization configuration optimized for Google Colab
    """
    if quantize_mode == "none":
        return None
    elif quantize_mode == "8bit":
        print("🔧 Using 8-bit quantization for Colab")
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True  # Colab-friendly CPU offload
        )
    elif quantize_mode == "4bit":
        print("🔧 Using 4-bit quantization for Colab (maximum memory savings)")
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,  # Double quantization for extra savings
            bnb_4bit_quant_type="nf4",       # NF4 is optimal for neural networks
            bnb_4bit_compute_dtype=torch.bfloat16,  # Colab T4 supports bfloat16
        )
    else:
        raise ValueError(f"Unknown quantization mode: {quantize_mode}")

def optimize_for_colab():
    """Apply Colab-specific optimizations"""
    print("🚀 Applying Google Colab optimizations...")
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print("   ✅ GPU cache cleared")
    
    # Enable memory-efficient attention if available
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        print("   ✅ Flash Attention enabled")
    except:
        print("   ⚠️ Flash Attention not available")
    
    # Set memory fraction for Colab T4 (15GB)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of available memory
        print("   ✅ GPU memory fraction set to 95%")

def test_quantization_config():
    """Test the quantization configuration function"""
    print("🧪 === TESTING COLAB QUANTIZATION CONFIG ===")
    
    # Test all quantization modes
    for mode in ["none", "8bit", "4bit"]:
        print(f"\n📦 Testing {mode} quantization...")
        try:
            config = get_colab_quantization_config(mode)
            if config is None:
                print(f"   ✅ {mode}: No quantization (as expected)")
            else:
                print(f"   ✅ {mode}: Config created successfully")
                print(f"      load_in_8bit: {getattr(config, 'load_in_8bit', False)}")
                print(f"      load_in_4bit: {getattr(config, 'load_in_4bit', False)}")
                if hasattr(config, 'bnb_4bit_quant_type'):
                    print(f"      quant_type: {config.bnb_4bit_quant_type}")
        except Exception as e:
            print(f"   ❌ {mode}: Failed with error: {e}")
    
    print(f"\n🚀 Testing Colab optimizations...")
    try:
        optimize_for_colab()
        print(f"   ✅ Colab optimizations applied successfully")
    except Exception as e:
        print(f"   ❌ Colab optimizations failed: {e}")

if __name__ == "__main__":
    test_quantization_config()
    print(f"\n🎯 Quantization configuration test complete!")
    print(f"\nTo run with quantization in Colab:")
    print(f"  python main.py --model_path llava --quantize 4bit --colab-mode")
    print(f"  python main.py --model_path qwen --quantize 8bit --colab-mode")
