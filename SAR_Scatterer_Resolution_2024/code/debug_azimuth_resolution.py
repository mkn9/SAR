#!/usr/bin/env python3
"""
Debug Azimuth Resolution - Investigate why plots show single targets when algorithm reports multiple detections
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append('../../SAR_Benchmark_Validation_2024/code')
from sar_model_final import FinalSARModel
from scipy.signal import find_peaks

def debug_range_resolved_case():
    """Debug the range resolved case to see why only one target appears"""
    print("DEBUGGING RANGE RESOLVED CASE")
    print("="*50)
    
    # Initialize SAR model
    sar = FinalSARModel(fc=10e9, B=100e6, Tp=10e-6)
    fs = 200e6
    c = 3e8
    range_resolution = c / (2 * 100e6)
    
    print(f"Range resolution: {range_resolution:.3f} m")
    
    # Define targets - 3m separation (should resolve)
    targets = [
        {'range': 1000.0, 'rcs': 1.0, 'name': 'Target_A'},
        {'range': 1003.0, 'rcs': 1.0, 'name': 'Target_B'}
    ]
    
    print(f"Target separation: {targets[1]['range'] - targets[0]['range']:.1f} m")
    print(f"Separation vs resolution: {(targets[1]['range'] - targets[0]['range']) / range_resolution:.1f}x")
    
    # Generate individual responses
    print("\nGenerating individual target responses...")
    t1, response1 = sar.point_target_response(R0=targets[0]['range'], fs=fs, plot=False)
    t2, response2 = sar.point_target_response(R0=targets[1]['range'], fs=fs, plot=False)
    
    # Find peaks in individual responses
    compressed1 = sar.range_compression(response1, plot=False)
    compressed2 = sar.range_compression(response2, plot=False)
    
    range_axis1 = t1 * c / 2
    range_axis2 = t2 * c / 2
    
    peak1_idx = np.argmax(np.abs(compressed1))
    peak2_idx = np.argmax(np.abs(compressed2))
    
    peak1_range = range_axis1[peak1_idx]
    peak2_range = range_axis2[peak2_idx]
    
    print(f"Individual target peaks:")
    print(f"  Target A: {peak1_range:.1f} m (expected {targets[0]['range']:.1f} m)")
    print(f"  Target B: {peak2_range:.1f} m (expected {targets[1]['range']:.1f} m)")
    
    # Generate combined response
    print("\nGenerating combined response...")
    min_len = min(len(response1), len(response2))
    combined_response = response1[:min_len] + response2[:min_len]
    combined_compressed = sar.range_compression(combined_response, plot=False)
    
    # Ensure same length for analysis
    min_analysis_len = min(len(t1), len(combined_compressed))
    range_axis = t1[:min_analysis_len] * c / 2
    magnitude = np.abs(combined_compressed[:min_analysis_len])
    
    # Peak detection with different thresholds
    print("\nTesting peak detection with different thresholds:")
    
    for threshold in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
        peaks_idx, properties = find_peaks(
            magnitude,
            height=threshold * np.max(magnitude),
            distance=max(1, int(range_resolution / (c/2) * fs))
        )
        
        detected_ranges = range_axis[peaks_idx]
        print(f"  Threshold {threshold:.1f}: {len(peaks_idx)} peaks at {[f'{r:.1f}' for r in detected_ranges]} m")
    
    # Create detailed diagnostic plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Debug: Range Resolution Analysis', fontsize=16)
    
    # Plot 1: Individual responses
    ax1 = axes[0, 0]
    ax1.plot(range_axis1, np.abs(compressed1), 'b-', label='Target A (1000m)', alpha=0.7)
    ax1.plot(range_axis2, np.abs(compressed2), 'r-', label='Target B (1003m)', alpha=0.7)
    ax1.axvline(1000.0, color='blue', linestyle='--', alpha=0.5, label='Expected A')
    ax1.axvline(1003.0, color='red', linestyle='--', alpha=0.5, label='Expected B')
    ax1.set_xlim(995, 1010)
    ax1.set_title('Individual Target Responses')
    ax1.set_xlabel('Range (m)')
    ax1.set_ylabel('Magnitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Combined response
    ax2 = axes[0, 1]
    ax2.plot(range_axis, magnitude, 'g-', linewidth=2, label='Combined Response')
    
    # Test different peak detection thresholds
    for i, threshold in enumerate([0.3, 0.5, 0.7]):
        peaks_idx, _ = find_peaks(
            magnitude,
            height=threshold * np.max(magnitude),
            distance=max(1, int(range_resolution / (c/2) * fs))
        )
        
        colors = ['orange', 'purple', 'brown']
        for j, peak_idx in enumerate(peaks_idx):
            ax2.plot(range_axis[peak_idx], magnitude[peak_idx], 'o', 
                    color=colors[i], markersize=8,
                    label=f'Thresh {threshold:.1f}' if j == 0 else '')
    
    ax2.axvline(1000.0, color='blue', linestyle='--', alpha=0.8, label='Expected A')
    ax2.axvline(1003.0, color='red', linestyle='--', alpha=0.8, label='Expected B')
    ax2.set_xlim(995, 1010)
    ax2.set_title('Combined Response with Peak Detection')
    ax2.set_xlabel('Range (m)')
    ax2.set_ylabel('Magnitude')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Zoomed view around peaks
    ax3 = axes[1, 0]
    ax3.plot(range_axis, magnitude, 'g-', linewidth=2)
    
    # Find best peaks
    peaks_idx, _ = find_peaks(
        magnitude,
        height=0.3 * np.max(magnitude),
        distance=max(1, int(range_resolution / (c/2) * fs))
    )
    
    for i, peak_idx in enumerate(peaks_idx):
        peak_range = range_axis[peak_idx]
        peak_mag = magnitude[peak_idx]
        ax3.plot(peak_range, peak_mag, 'ro', markersize=10, 
                label=f'Peak {i+1}: {peak_range:.1f}m')
        ax3.text(peak_range, peak_mag + 0.1*np.max(magnitude), 
                f'{peak_range:.1f}m', ha='center', fontsize=10)
    
    ax3.axvline(1000.0, color='blue', linestyle='--', alpha=0.8)
    ax3.axvline(1003.0, color='red', linestyle='--', alpha=0.8)
    ax3.set_xlim(998, 1006)
    ax3.set_title(f'Detected Peaks: {len(peaks_idx)} peaks')
    ax3.set_xlabel('Range (m)')
    ax3.set_ylabel('Magnitude')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Resolution analysis
    ax4 = axes[1, 1]
    
    # Calculate theoretical resolution limit
    resolution_samples = range_resolution / (c/2) * fs
    
    ax4.text(0.1, 0.8, f'Range Resolution: {range_resolution:.3f} m', transform=ax4.transAxes, fontsize=12)
    ax4.text(0.1, 0.7, f'Target Separation: {targets[1]["range"] - targets[0]["range"]:.1f} m', transform=ax4.transAxes, fontsize=12)
    ax4.text(0.1, 0.6, f'Separation/Resolution: {(targets[1]["range"] - targets[0]["range"]) / range_resolution:.1f}x', transform=ax4.transAxes, fontsize=12)
    ax4.text(0.1, 0.5, f'Resolution in Samples: {resolution_samples:.1f}', transform=ax4.transAxes, fontsize=12)
    ax4.text(0.1, 0.4, f'Detected Peaks: {len(peaks_idx)}', transform=ax4.transAxes, fontsize=12, 
            color='green' if len(peaks_idx) == 2 else 'red')
    
    if len(peaks_idx) >= 2:
        peak_separation = abs(range_axis[peaks_idx[1]] - range_axis[peaks_idx[0]])
        ax4.text(0.1, 0.3, f'Peak Separation: {peak_separation:.1f} m', transform=ax4.transAxes, fontsize=12)
        resolution_success = peak_separation > 0.5 * range_resolution
        ax4.text(0.1, 0.2, f'Resolution Success: {resolution_success}', transform=ax4.transAxes, fontsize=12,
                color='green' if resolution_success else 'red')
    
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_title('Resolution Analysis')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig('../output/debug_range_resolution_analysis.png', dpi=300, bbox_inches='tight')
    print("\nDiagnostic plot saved to: ../output/debug_range_resolution_analysis.png")
    plt.close()
    
    # Summary
    print(f"\n" + "="*50)
    print("DIAGNOSIS SUMMARY")
    print("="*50)
    print(f"Expected: 2 targets separated by {targets[1]['range'] - targets[0]['range']:.1f} m")
    print(f"Detected: {len(peaks_idx)} peaks")
    
    if len(peaks_idx) == 2:
        peak_separation = abs(range_axis[peaks_idx[1]] - range_axis[peaks_idx[0]])
        print(f"Peak separation: {peak_separation:.1f} m")
        print("✅ RESOLUTION SUCCESSFUL - Two peaks detected")
    elif len(peaks_idx) == 1:
        print("❌ RESOLUTION FAILED - Only one peak detected")
        print("Possible causes:")
        print("  - Peak detection threshold too high")
        print("  - Targets too close relative to SAR pulse width")
        print("  - Signal processing artifacts")
    else:
        print(f"⚠️  UNEXPECTED - {len(peaks_idx)} peaks detected")
    
    return len(peaks_idx) == 2

if __name__ == "__main__":
    success = debug_range_resolved_case()
    print(f"\nRange resolution test: {'PASSED' if success else 'FAILED'}")
