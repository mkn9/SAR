#!/usr/bin/env python3
"""
Test range axis calculation to debug the range detection issue
"""

import numpy as np
import sys
import os
sys.path.append('../../SAR_Benchmark_Validation_2024/code')
from sar_model_final import FinalSARModel

def test_range_axis():
    """Test range axis calculation against known target"""
    print("TESTING RANGE AXIS CALCULATION")
    print("="*50)
    
    # Initialize SAR model
    sar = FinalSARModel(fc=10e9, B=100e6, Tp=10e-6)
    fs = 200e6  # 200 MHz sampling
    c = 3e8
    
    # Test target at 1000m
    target_range = 1000.0
    print(f"Target at: {target_range} m")
    
    # Generate target response using SAR model method
    t, response = sar.point_target_response(R0=target_range, fs=fs, plot=False)
    
    # Range compress
    compressed = sar.range_compression(response, plot=False)
    
    # Find peak
    magnitude = np.abs(compressed)
    peak_idx = np.argmax(magnitude)
    
    # Calculate range axis multiple ways
    print(f"\nRange axis calculations:")
    print(f"Signal length: {len(magnitude)} samples")
    print(f"Sampling frequency: {fs/1e6} MHz")
    print(f"Peak at sample: {peak_idx}")
    
    # Method 1: Direct from time vector
    peak_time = t[peak_idx]
    detected_range_1 = peak_time * c / 2
    print(f"Method 1 (from time vector): {detected_range_1:.1f} m")
    
    # Method 2: From sample index
    dt = 1/fs
    peak_time_2 = peak_idx * dt
    detected_range_2 = peak_time_2 * c / 2
    print(f"Method 2 (from sample index): {detected_range_2:.1f} m")
    
    # Method 3: Range axis array
    range_axis = t * c / 2
    detected_range_3 = range_axis[peak_idx]
    print(f"Method 3 (range axis array): {detected_range_3:.1f} m")
    
    # Check errors
    error_1 = abs(detected_range_1 - target_range)
    error_2 = abs(detected_range_2 - target_range)
    error_3 = abs(detected_range_3 - target_range)
    
    print(f"\nRange errors:")
    print(f"Method 1 error: {error_1:.1f} m")
    print(f"Method 2 error: {error_2:.1f} m")
    print(f"Method 3 error: {error_3:.1f} m")
    
    # Test the experiment's range axis calculation
    print(f"\nExperiment range axis test:")
    range_samples = len(magnitude)
    time_axis_exp = np.arange(range_samples) / fs
    range_axis_exp = time_axis_exp * c / 2
    detected_range_exp = range_axis_exp[peak_idx]
    error_exp = abs(detected_range_exp - target_range)
    
    print(f"Experiment method: {detected_range_exp:.1f} m")
    print(f"Experiment error: {error_exp:.1f} m")
    
    # Summary
    print(f"\n" + "="*50)
    if error_exp < 10:
        print("✅ Range axis calculation is CORRECT")
    else:
        print("❌ Range axis calculation has ERROR")
        print("Issue may be in data simulation or processing chain")
    
    return error_exp < 10

if __name__ == "__main__":
    success = test_range_axis()
