#!/usr/bin/env python3
"""
Debug SAR Scatterer Resolution Experiment
Investigates and fixes the peak detection and plotting issues
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append('../../SAR_Benchmark_Validation_2024/code')
from sar_model_final import FinalSARModel
from scipy.signal import find_peaks
import json

class DebugScattererExperiment:
    def __init__(self, fc=10e9, B=100e6, Tp=10e-6):
        """Initialize with simpler parameters for debugging"""
        self.fc = fc
        self.B = B
        self.Tp = Tp
        self.c = 3e8
        
        # Initialize SAR model
        self.sar = FinalSARModel(fc=fc, B=B, Tp=Tp)
        
        # Calculate system parameters
        self.wavelength = self.c / fc
        self.range_resolution = self.c / (2 * B)
        self.fs = 2 * B  # Nyquist sampling frequency
        
        print(f"Debug SAR Experiment Initialized:")
        print(f"Range Resolution: {self.range_resolution:.3f} m")
        print(f"Sampling Frequency: {self.fs/1e6:.1f} MHz")
    
    def debug_single_target(self, target_range=1000):
        """Debug with single target to understand signal characteristics"""
        print(f"\n=== DEBUGGING SINGLE TARGET AT {target_range}m ===")
        
        # Generate single target response
        t, response = self.sar.point_target_response(R0=target_range, fs=self.fs, plot=False)
        
        # Range compress
        compressed = self.sar.range_compression(response, plot=False)
        
        # Analyze signal
        magnitude = np.abs(compressed)
        range_axis = t * self.c / 2  # Convert time to range
        
        print(f"Signal length: {len(magnitude)} samples")
        print(f"Range axis: {range_axis[0]:.1f} to {range_axis[-1]:.1f} m")
        print(f"Max magnitude: {np.max(magnitude):.2e}")
        print(f"Mean magnitude: {np.mean(magnitude):.2e}")
        
        # Find peak
        peak_idx = np.argmax(magnitude)
        detected_range = range_axis[peak_idx]
        range_error = abs(detected_range - target_range)
        
        print(f"Peak at index: {peak_idx}")
        print(f"Detected range: {detected_range:.1f} m")
        print(f"Range error: {range_error:.1f} m")
        
        # Plot for visualization
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(t*1e6, np.abs(response), 'b-', linewidth=1)
        plt.title(f'Target Response (R0={target_range}m)')
        plt.xlabel('Time (μs)')
        plt.ylabel('Magnitude')
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(range_axis, magnitude, 'r-', linewidth=1)
        plt.axvline(target_range, color='g', linestyle='--', label=f'Expected ({target_range}m)')
        plt.axvline(detected_range, color='k', linestyle='-', label=f'Detected ({detected_range:.1f}m)')
        plt.title('Range Compressed Signal')
        plt.xlabel('Range (m)')
        plt.ylabel('Magnitude')
        plt.legend()
        plt.grid(True)
        
        # Zoom in around target
        plt.subplot(2, 2, 3)
        zoom_range = 50  # meters
        zoom_mask = (range_axis >= target_range - zoom_range) & (range_axis <= target_range + zoom_range)
        plt.plot(range_axis[zoom_mask], magnitude[zoom_mask], 'r-', linewidth=2)
        plt.axvline(target_range, color='g', linestyle='--', label=f'Expected')
        plt.axvline(detected_range, color='k', linestyle='-', label=f'Detected')
        plt.title('Zoomed View Around Target')
        plt.xlabel('Range (m)')
        plt.ylabel('Magnitude')
        plt.legend()
        plt.grid(True)
        
        # Signal statistics
        plt.subplot(2, 2, 4)
        plt.semilogy(range_axis, magnitude, 'r-', linewidth=1)
        plt.axvline(target_range, color='g', linestyle='--')
        plt.axvline(detected_range, color='k', linestyle='-')
        plt.title('Signal Magnitude (Log Scale)')
        plt.xlabel('Range (m)')
        plt.ylabel('Magnitude (log)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('../output/debug_single_target.png', dpi=300, bbox_inches='tight')
        print("Debug plot saved to: ../output/debug_single_target.png")
        plt.close()
        
        return range_axis, magnitude, detected_range, range_error
    
    def debug_two_targets(self, R1=1000, R2=1002):
        """Debug two-target scenario to understand resolution issues"""
        print(f"\n=== DEBUGGING TWO TARGETS: {R1}m and {R2}m ===")
        separation = abs(R2 - R1)
        print(f"Target separation: {separation:.1f} m")
        print(f"Range resolution: {self.range_resolution:.3f} m")
        print(f"Separation ratio: {separation/self.range_resolution:.2f}x resolution")
        
        # Generate individual target responses
        t1, response1 = self.sar.point_target_response(R0=R1, fs=self.fs, plot=False)
        t2, response2 = self.sar.point_target_response(R0=R2, fs=self.fs, plot=False)
        
        # Ensure same length
        min_len = min(len(response1), len(response2))
        response1 = response1[:min_len]
        response2 = response2[:min_len]
        t = t1[:min_len]
        
        # Combine responses
        combined_response = response1 + response2
        
        # Range compress
        compressed = self.sar.range_compression(combined_response, plot=False)
        compressed = compressed[:min_len]
        
        # Analyze
        magnitude = np.abs(compressed)
        range_axis = t * self.c / 2
        
        print(f"Combined signal length: {len(magnitude)} samples")
        print(f"Range axis: {range_axis[0]:.1f} to {range_axis[-1]:.1f} m")
        print(f"Max magnitude: {np.max(magnitude):.2e}")
        
        # Advanced peak detection
        # Try different thresholds
        thresholds = [0.1, 0.2, 0.3, 0.5, 0.7]
        
        print(f"\nPeak detection analysis:")
        for threshold in thresholds:
            peaks_idx, properties = find_peaks(
                magnitude,
                height=threshold * np.max(magnitude),
                distance=int(self.range_resolution / (self.c/2) * self.fs)  # Resolution distance in samples
            )
            
            if len(peaks_idx) > 0:
                detected_ranges = range_axis[peaks_idx]
                print(f"Threshold {threshold*100:3.0f}%: {len(peaks_idx):2d} peaks at {detected_ranges[:5]}")  # Show first 5
            else:
                print(f"Threshold {threshold*100:3.0f}%:  0 peaks")
        
        # Use reasonable threshold for detailed analysis
        peaks_idx, properties = find_peaks(
            magnitude,
            height=0.3 * np.max(magnitude),
            distance=int(self.range_resolution / (self.c/2) * self.fs)
        )
        
        detected_ranges = range_axis[peaks_idx] if len(peaks_idx) > 0 else []
        
        print(f"\nUsing 30% threshold:")
        print(f"Detected peaks: {len(peaks_idx)}")
        if len(detected_ranges) > 0:
            print(f"Peak ranges: {detected_ranges}")
            
            # Calculate errors
            expected_ranges = [R1, R2]
            if len(detected_ranges) >= 2:
                error1 = min(abs(detected_ranges[0] - R1), abs(detected_ranges[0] - R2))
                error2 = min(abs(detected_ranges[1] - R1), abs(detected_ranges[1] - R2))
                print(f"Range errors: {error1:.1f}m, {error2:.1f}m")
                
                # Check if targets are resolved
                peak_separation = abs(detected_ranges[1] - detected_ranges[0])
                resolved = peak_separation >= 0.5 * self.range_resolution
                print(f"Peak separation: {peak_separation:.3f} m")
                print(f"Resolution status: {'RESOLVED' if resolved else 'UNRESOLVED'}")
        
        # Create detailed plot
        plt.figure(figsize=(15, 10))
        
        # Individual responses
        plt.subplot(2, 3, 1)
        plt.plot(range_axis, np.abs(response1), 'b-', label=f'Target 1 ({R1}m)', alpha=0.7)
        plt.plot(range_axis, np.abs(response2), 'r-', label=f'Target 2 ({R2}m)', alpha=0.7)
        plt.axvline(R1, color='b', linestyle='--', alpha=0.5)
        plt.axvline(R2, color='r', linestyle='--', alpha=0.5)
        plt.title('Individual Target Responses')
        plt.xlabel('Range (m)')
        plt.ylabel('Magnitude')
        plt.legend()
        plt.grid(True)
        
        # Combined response
        plt.subplot(2, 3, 2)
        plt.plot(range_axis, np.abs(combined_response), 'g-', linewidth=1)
        plt.axvline(R1, color='b', linestyle='--', alpha=0.7, label=f'Target 1')
        plt.axvline(R2, color='r', linestyle='--', alpha=0.7, label=f'Target 2')
        plt.title('Combined Target Response')
        plt.xlabel('Range (m)')
        plt.ylabel('Magnitude')
        plt.legend()
        plt.grid(True)
        
        # Range compressed
        plt.subplot(2, 3, 3)
        plt.plot(range_axis, magnitude, 'k-', linewidth=1)
        plt.axvline(R1, color='b', linestyle='--', alpha=0.7, label=f'Expected 1')
        plt.axvline(R2, color='r', linestyle='--', alpha=0.7, label=f'Expected 2')
        
        # Mark detected peaks
        if len(peaks_idx) > 0:
            for i, (idx, range_val) in enumerate(zip(peaks_idx[:4], detected_ranges[:4])):  # Show first 4
                plt.plot(range_val, magnitude[idx], 'go', markersize=8, 
                        label=f'Peak {i+1}' if i < 2 else '')
        
        plt.title('Range Compressed Signal')
        plt.xlabel('Range (m)')
        plt.ylabel('Magnitude')
        plt.legend()
        plt.grid(True)
        
        # Zoomed view around targets
        plt.subplot(2, 3, 4)
        zoom_range = 10  # meters
        center_range = (R1 + R2) / 2
        zoom_mask = (range_axis >= center_range - zoom_range) & (range_axis <= center_range + zoom_range)
        
        plt.plot(range_axis[zoom_mask], magnitude[zoom_mask], 'k-', linewidth=2)
        plt.axvline(R1, color='b', linestyle='--', linewidth=2, label=f'Target 1')
        plt.axvline(R2, color='r', linestyle='--', linewidth=2, label=f'Target 2')
        
        # Mark detected peaks in zoom
        if len(peaks_idx) > 0:
            for idx, range_val in zip(peaks_idx, detected_ranges):
                if center_range - zoom_range <= range_val <= center_range + zoom_range:
                    plt.plot(range_val, magnitude[idx], 'go', markersize=10)
        
        plt.title(f'Zoomed View (±{zoom_range}m)')
        plt.xlabel('Range (m)')
        plt.ylabel('Magnitude')
        plt.legend()
        plt.grid(True)
        
        # Peak detection threshold analysis
        plt.subplot(2, 3, 5)
        for threshold in [0.1, 0.3, 0.5, 0.7]:
            threshold_line = threshold * np.max(magnitude) * np.ones_like(range_axis)
            plt.plot(range_axis[zoom_mask], threshold_line[zoom_mask], '--', 
                    label=f'{threshold*100:.0f}% threshold', alpha=0.7)
        
        plt.plot(range_axis[zoom_mask], magnitude[zoom_mask], 'k-', linewidth=2, label='Signal')
        plt.axvline(R1, color='b', linestyle=':', alpha=0.7)
        plt.axvline(R2, color='r', linestyle=':', alpha=0.7)
        plt.title('Threshold Analysis')
        plt.xlabel('Range (m)')
        plt.ylabel('Magnitude')
        plt.legend()
        plt.grid(True)
        
        # Resolution analysis
        plt.subplot(2, 3, 6)
        # Show theoretical resolution limit
        resolution_samples = int(self.range_resolution / (self.c/2) * self.fs)
        
        plt.plot(range_axis[zoom_mask], magnitude[zoom_mask], 'k-', linewidth=2)
        plt.axvline(R1, color='b', linestyle='--', linewidth=2)
        plt.axvline(R2, color='r', linestyle='--', linewidth=2)
        
        # Show resolution cell
        plt.axvspan(R1 - self.range_resolution/2, R1 + self.range_resolution/2, 
                   alpha=0.2, color='blue', label='Resolution cell 1')
        plt.axvspan(R2 - self.range_resolution/2, R2 + self.range_resolution/2, 
                   alpha=0.2, color='red', label='Resolution cell 2')
        
        plt.title(f'Resolution Analysis\nSeparation: {separation:.1f}m vs Resolution: {self.range_resolution:.1f}m')
        plt.xlabel('Range (m)')
        plt.ylabel('Magnitude')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('../output/debug_two_targets.png', dpi=300, bbox_inches='tight')
        print("Debug plot saved to: ../output/debug_two_targets.png")
        plt.close()
        
        return detected_ranges, len(peaks_idx)
    
    def run_debug_analysis(self):
        """Run complete debug analysis"""
        print("SAR SCATTERER RESOLUTION DEBUG ANALYSIS")
        print("="*60)
        
        # Create output directory
        os.makedirs('../output', exist_ok=True)
        
        # Debug single target first
        range_axis, magnitude, detected_range, error = self.debug_single_target(1000)
        
        # Debug two-target scenarios
        print("\n" + "="*60)
        print("TWO-TARGET RESOLUTION ANALYSIS")
        print("="*60)
        
        scenarios = [
            {'R1': 1000, 'R2': 1000.8, 'description': 'Unresolved (0.8m separation)'},
            {'R1': 1000, 'R2': 1002.0, 'description': 'Resolved (2.0m separation)'},
            {'R1': 1000, 'R2': 1003.0, 'description': 'Well resolved (3.0m separation)'}
        ]
        
        for scenario in scenarios:
            print(f"\n--- {scenario['description']} ---")
            detected_ranges, num_peaks = self.debug_two_targets(scenario['R1'], scenario['R2'])
        
        print(f"\nDebug analysis complete! Check ../output/ for diagnostic plots.")

def main():
    """Main debug execution"""
    debug_experiment = DebugScattererExperiment()
    debug_experiment.run_debug_analysis()

if __name__ == "__main__":
    main()
