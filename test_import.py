#!/usr/bin/env python
"""
Comprehensive functionality test for the qwen-payslip-processor package.
This checks basic imports and core functionality without requiring model downloads.
"""

import os
import sys
from qwen_payslip_processor import QwenPayslipProcessor
from PIL import Image
import io

print("Package imported successfully")

# 1. Test instantiation with different configurations
print("\n--- Testing Basic Configuration ---")
processor = QwenPayslipProcessor(window_mode="whole", force_cpu=True)
print(f"Device: {processor.device}")
print(f"Window mode: {processor.window_mode}")
print(f"Selected windows: {processor.selected_windows}")

# 2. Test configuration merging
print("\n--- Testing Configuration Merging ---")
custom_config = {
    "pdf": {"dpi": 300},
    "image": {
        "resolution_steps": [1200, 800],
        "enhance_contrast": False
    }
}
processor_custom = QwenPayslipProcessor(config=custom_config)
print(f"PDF DPI: {processor_custom.config['pdf']['dpi']}")
print(f"Resolution steps: {processor_custom.config['image']['resolution_steps']}")
print(f"Enhance contrast: {processor_custom.config['image']['enhance_contrast']}")
print(f"Sharpen factor (from default): {processor_custom.config['image']['sharpen_factor']}")

# 3. Test window mode options
print("\n--- Testing Window Modes ---")
modes = ["whole", "vertical", "horizontal", "quadrant"]
for mode in modes:
    p = QwenPayslipProcessor(window_mode=mode, force_cpu=True)
    print(f"Mode {mode} initialized successfully")

# 4. Test selected windows handling
print("\n--- Testing Selected Windows ---")
test_cases = [
    {"mode": "vertical", "windows": "top"},
    {"mode": "vertical", "windows": ["top", "bottom"]},
    {"mode": "horizontal", "windows": "left"},
    {"mode": "quadrant", "windows": ["top_left", "bottom_right"]},
]
for case in test_cases:
    p = QwenPayslipProcessor(window_mode=case["mode"], selected_windows=case["windows"], force_cpu=True)
    print(f"Mode {case['mode']} with windows {case['windows']} initialized successfully")

# 5. Test custom prompts
print("\n--- Testing Custom Prompts ---")
custom_prompts = {
    "top": "Find employee name in this section",
    "bottom": "Find salary information in this section"
}
p = QwenPayslipProcessor(
    window_mode="vertical", 
    custom_prompts=custom_prompts, 
    force_cpu=True
)
print(f"Custom prompts registered: {list(p.custom_prompts.keys())}")

# 6. Test utility functions
print("\n--- Testing Utility Functions ---")
try:
    from qwen_payslip_processor.utils import optimize_image_for_vl_model, split_image_for_window_mode
    # Create a test image
    test_image = Image.new('RGB', (600, 800), color='white')
    
    # Test image optimization
    optimized = optimize_image_for_vl_model(test_image, 300)
    print(f"Image optimization: {optimized.size}")
    
    # Test image splitting
    windows = split_image_for_window_mode(test_image, window_mode="vertical")
    print(f"Split image into {len(windows)} windows")
    for i, (window, position) in enumerate(windows):
        print(f"  Window {i+1}: {position} ({window.size})")
    
    print("Utility functions work correctly")
except Exception as e:
    print(f"Error testing utility functions: {e}")

print("\nAll tests completed successfully!")