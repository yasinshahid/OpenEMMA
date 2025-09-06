#!/usr/bin/env python3
"""
Quick debug test script to validate our debugging functions
Run this after installing dependencies to test the debugging system
"""

import torch
import sys
import os

# Add current directory to path to import our debugging functions
sys.path.append('.')

def test_debug_functions():
    """Test our debugging functions"""
    print("🧪 Testing debugging functions...")
    
    try:
        # Test GPU memory debugging
        from main import debug_gpu_memory
        debug_gpu_memory("Test Run")
        
        # Test tensor debugging with a simple model
        if torch.cuda.is_available():
            # Create a simple model for testing
            model = torch.nn.Linear(10, 1)
            model = model.cuda()
            
            from main import debug_model_tensors
            debug_model_tensors(model, "Test Model")
            
            print("✅ All debugging functions work!")
        else:
            print("⚠️ No CUDA available, skipping GPU tests")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you're running this after installing all dependencies")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    test_debug_functions()
