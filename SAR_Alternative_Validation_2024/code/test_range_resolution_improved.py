#!/usr/bin/env python3
"""
Improved Range Resolution Validation Tests
Enhanced peak detection algorithm for better two-point target separation testing
Based on fundamental radar theory: ΔR = c/(2*B)
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append('../../SAR_Benchmark_Validation_2024/code')
from sar_model_final import FinalSARModel
from scipy.signal import find_peaks
import unittest

class ImprovedRangeResolutionTests:
    def __init__(self):
        self.c = 3e8  # Speed of light
        self.test_results = {}
        
    def advanced_peak_detection(self, magnitude, min_height_ratio=0.1, min_distance_samples=10):
        """
        Advanced peak detection using scipy.signal.find_peaks with improved parameters
        
        Args:
            magnitude: Signal magnitude array
            min_height_ratio: Minimum peak height as ratio of maximum (default 0.1 = 10%)
            min_distance_samples: Minimum distance between peaks in samples
            
        Returns:
            peaks_idx: Array of peak indices
            properties: Dictionary with peak properties
        """
        # Calculate dynamic threshold
        max_mag = np.max(magnitude)
        min_height = min_height_ratio * max_mag
        
        # Find peaks with scipy
        peaks_idx, properties = find_peaks(
            magnitude,
            height=min_height,           # Minimum height threshold
            distance=min_distance_samples, # Minimum distance between peaks
            prominence=0.05 * max_mag,   # Minimum prominence (5% of max)
            width=1                      # Minimum width
        )
        
        return peaks_idx, properties
    
    def analyze_peak_separation(self, magnitude, peaks_idx, time_axis):
        """
        Analyze separation between detected peaks
        
        Args:
            magnitude: Signal magnitude array
            peaks_idx: Array of peak indices
            time_axis: Time axis for range conversion
            
        Returns:
            analysis: Dictionary with separation analysis
        """
        if len(peaks_idx) < 2:
            return {
                'num_peaks': len(peaks_idx),
                'separation_m': None,
                'valley_depth_db': None,
                'resolved': False,
                'peak_positions_m': []
            }
        
        # Convert to range positions
        range_axis = time_axis * self.c / 2
        peak_positions = [range_axis[idx] for idx in peaks_idx]
        
        # Calculate separation between first two peaks
        separation = abs(peak_positions[1] - peak_positions[0])
        
        # Find valley between first two peaks
        start_idx = min(peaks_idx[0], peaks_idx[1])
        end_idx = max(peaks_idx[0], peaks_idx[1])
        valley_idx = start_idx + np.argmin(magnitude[start_idx:end_idx+1])
        
        # Calculate valley depth in dB
        peak_max = max(magnitude[peaks_idx[0]], magnitude[peaks_idx[1]])
        valley_min = magnitude[valley_idx]
        valley_depth_db = 20 * np.log10(valley_min / peak_max)
        
        # Apply 3dB resolution criterion
        resolved = valley_depth_db < -3.0
        
        return {
            'num_peaks': len(peaks_idx),
            'separation_m': separation,
            'valley_depth_db': valley_depth_db,
            'resolved': resolved,
            'peak_positions_m': peak_positions,
            'peak_magnitudes': [magnitude[idx] for idx in peaks_idx],
            'valley_position_m': range_axis[valley_idx],
            'valley_magnitude': valley_min
        }
    
    def test_two_point_target_separation_improved(self):
        """Test 2: Improved Two-Point Target Separation Test"""
        print("\n" + "="*60)
        print("TEST 2: IMPROVED TWO-POINT TARGET SEPARATION TEST")
        print("="*60)
        
        # Test parameters
        fc = 10e9  # 10 GHz
        B = 100e6  # 100 MHz
        Tp = 10e-6
        
        sar = FinalSARModel(fc=fc, B=B, Tp=Tp)
        theoretical_res = sar.calculate_range_resolution()
        
        print(f"Theoretical Range Resolution: {theoretical_res:.3f} m")
        print(f"Expected separation for resolution limit: {theoretical_res:.3f} m")
        
        # Enhanced test cases with more scenarios
        test_cases = [
            {'R1': 1000, 'R2': 1000 + theoretical_res*0.6, 'description': 'Well below limit (0.6x)'},
            {'R1': 1000, 'R2': 1000 + theoretical_res*0.8, 'description': 'Below limit (0.8x)'},
            {'R1': 1000, 'R2': 1000 + theoretical_res*1.0, 'description': 'At resolution limit (1.0x)'},
            {'R1': 1000, 'R2': 1000 + theoretical_res*1.2, 'description': 'Above limit (1.2x)'},
            {'R1': 1000, 'R2': 1000 + theoretical_res*1.5, 'description': 'Well above limit (1.5x)'},
            {'R1': 1000, 'R2': 1000 + theoretical_res*2.0, 'description': 'Far above limit (2.0x)'}
        ]
        
        results = []
        
        for i, case in enumerate(test_cases):
            print(f"\n--- Case {i+1}: {case['description']} ---")
            print(f"Target 1: {case['R1']:.1f} m")
            print(f"Target 2: {case['R2']:.1f} m") 
            print(f"Separation: {case['R2'] - case['R1']:.3f} m")
            print(f"Ratio to resolution: {(case['R2'] - case['R1'])/theoretical_res:.2f}x")
            
            # Generate individual target responses
            t1, response1 = sar.point_target_response(R0=case['R1'], plot=False)
            t2, response2 = sar.point_target_response(R0=case['R2'], plot=False)
            
            # Ensure responses have same length and time base
            min_len = min(len(response1), len(response2))
            response1 = response1[:min_len]
            response2 = response2[:min_len]
            t_combined = t1[:min_len]
            
            # Combine responses (equal amplitude)
            combined_response = response1 + response2
            
            # Range compress
            compressed = sar.range_compression(combined_response, plot=False)
            compressed = compressed[:min_len]  # Ensure same length
            
            # Apply improved peak detection
            magnitude = np.abs(compressed)
            
            # Try multiple detection thresholds
            thresholds = [0.05, 0.1, 0.15, 0.2]  # 5%, 10%, 15%, 20%
            best_result = None
            
            for threshold in thresholds:
                peaks_idx, properties = self.advanced_peak_detection(
                    magnitude, 
                    min_height_ratio=threshold,
                    min_distance_samples=max(5, int(len(magnitude) * 0.01))  # Adaptive distance
                )
                
                if len(peaks_idx) >= 2:
                    analysis = self.analyze_peak_separation(magnitude, peaks_idx, t_combined)
                    
                    # Check if this gives reasonable separation
                    if analysis['separation_m'] is not None:
                        expected_sep = case['R2'] - case['R1']
                        sep_error = abs(analysis['separation_m'] - expected_sep) / expected_sep
                        
                        if sep_error < 0.5:  # Within 50% of expected
                            best_result = analysis
                            break
            
            # If no good result found, use the lowest threshold result
            if best_result is None:
                peaks_idx, properties = self.advanced_peak_detection(
                    magnitude, 
                    min_height_ratio=0.05,  # Very low threshold
                    min_distance_samples=3
                )
                best_result = self.analyze_peak_separation(magnitude, peaks_idx, t_combined)
            
            # Print results
            print(f"Peaks detected: {best_result['num_peaks']}")
            if best_result['separation_m'] is not None:
                print(f"Measured separation: {best_result['separation_m']:.3f} m")
                print(f"Valley depth: {best_result['valley_depth_db']:.1f} dB")
                print(f"Resolved (3dB criterion): {'YES' if best_result['resolved'] else 'NO'}")
                
                # Additional diagnostics
                expected_sep = case['R2'] - case['R1']
                sep_error = abs(best_result['separation_m'] - expected_sep) / expected_sep * 100
                print(f"Separation error: {sep_error:.1f}%")
                
            else:
                print("Insufficient peaks for separation analysis")
            
            results.append({
                'case': case['description'],
                'theoretical_sep': case['R2'] - case['R1'],
                'measured_sep': best_result['separation_m'],
                'valley_depth_db': best_result['valley_depth_db'],
                'resolved': best_result['resolved'],
                'peaks_found': best_result['num_peaks'],
                'ratio_to_resolution': (case['R2'] - case['R1'])/theoretical_res
            })
        
        # Assessment with improved criteria
        expected_results = {
            0: False,  # 0.6x - should not resolve
            1: False,  # 0.8x - should not resolve  
            2: True,   # 1.0x - should marginally resolve
            3: True,   # 1.2x - should resolve
            4: True,   # 1.5x - should clearly resolve
            5: True    # 2.0x - should clearly resolve
        }
        
        correct_predictions = 0
        total_predictions = 0
        
        print(f"\n--- SEPARATION TEST ASSESSMENT ---")
        for i, result in enumerate(results):
            if result['measured_sep'] is not None:
                expected = expected_results.get(i, None)
                actual = result['resolved']
                
                status = "✓" if expected == actual else "✗"
                print(f"Case {i+1}: Expected {'RESOLVE' if expected else 'NO RESOLVE'}, "
                      f"Got {'RESOLVE' if actual else 'NO RESOLVE'} {status}")
                
                if expected == actual:
                    correct_predictions += 1
                total_predictions += 1
            else:
                print(f"Case {i+1}: No separation measurement possible")
        
        # Overall test assessment
        if total_predictions > 0:
            accuracy = correct_predictions / total_predictions
            test_passed = accuracy >= 0.7  # 70% accuracy threshold
            print(f"\nPrediction Accuracy: {correct_predictions}/{total_predictions} ({accuracy*100:.1f}%)")
        else:
            test_passed = False
            print(f"\nNo valid separation measurements obtained")
        
        print(f"Test Result: {'PASS' if test_passed else 'FAIL'}")
        
        self.test_results['improved_two_point_separation'] = {
            'results': results,
            'accuracy': accuracy if total_predictions > 0 else 0,
            'passed': test_passed
        }
        
        return test_passed
    
    def create_diagnostic_plots(self, save_plots=True):
        """Create diagnostic plots for peak detection analysis"""
        
        if not save_plots:
            return
            
        # Test case for diagnostic plotting
        fc = 10e9  # 10 GHz
        B = 100e6  # 100 MHz
        Tp = 10e-6
        
        sar = FinalSARModel(fc=fc, B=B, Tp=Tp)
        theoretical_res = sar.calculate_range_resolution()
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Peak Detection Diagnostic Analysis', fontsize=16)
        
        test_cases = [
            {'R1': 1000, 'R2': 1000 + theoretical_res*0.8, 'title': 'Below Limit (0.8x)'},
            {'R1': 1000, 'R2': 1000 + theoretical_res*1.0, 'title': 'At Limit (1.0x)'},
            {'R1': 1000, 'R2': 1000 + theoretical_res*1.2, 'title': 'Above Limit (1.2x)'},
            {'R1': 1000, 'R2': 1000 + theoretical_res*2.0, 'title': 'Far Above Limit (2.0x)'}
        ]
        
        for idx, (case, ax) in enumerate(zip(test_cases, axes.flat)):
            # Generate responses
            t1, response1 = sar.point_target_response(R0=case['R1'], plot=False)
            t2, response2 = sar.point_target_response(R0=case['R2'], plot=False)
            
            min_len = min(len(response1), len(response2))
            combined_response = (response1 + response2)[:min_len]
            t_combined = t1[:min_len]
            
            # Range compress
            compressed = sar.range_compression(combined_response, plot=False)
            compressed = compressed[:min_len]
            magnitude = np.abs(compressed)
            
            # Convert to range axis
            range_axis = t_combined * self.c / 2
            
            # Detect peaks
            peaks_idx, properties = self.advanced_peak_detection(magnitude, min_height_ratio=0.1)
            
            # Plot
            ax.plot(range_axis, magnitude, 'b-', linewidth=1, label='Compressed Signal')
            ax.plot(range_axis[peaks_idx], magnitude[peaks_idx], 'ro', markersize=8, label=f'Peaks ({len(peaks_idx)})')
            
            # Mark theoretical target positions
            ax.axvline(case['R1'], color='g', linestyle='--', alpha=0.7, label='Target 1')
            ax.axvline(case['R2'], color='g', linestyle='--', alpha=0.7, label='Target 2')
            
            ax.set_title(f"{case['title']}\nSep: {case['R2']-case['R1']:.3f}m")
            ax.set_xlabel('Range (m)')
            ax.set_ylabel('Magnitude')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            
            # Set reasonable axis limits
            center_range = (case['R1'] + case['R2']) / 2
            ax.set_xlim(center_range - 5*theoretical_res, center_range + 5*theoretical_res)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('../output/peak_detection_diagnostics.png', dpi=300, bbox_inches='tight')
            print("Diagnostic plots saved to: ../output/peak_detection_diagnostics.png")
        
        plt.close()
    
    def run_improved_tests(self):
        """Run improved range resolution tests"""
        print("IMPROVED RANGE RESOLUTION TEST SUITE")
        print("="*60)
        
        # Create output directory
        os.makedirs('../output', exist_ok=True)
        
        # Run the improved two-point separation test
        test_passed = self.test_two_point_target_separation_improved()
        
        # Create diagnostic plots
        self.create_diagnostic_plots(save_plots=True)
        
        print("\n" + "="*60)
        print("IMPROVED RANGE RESOLUTION TEST SUMMARY")
        print("="*60)
        print(f"Improved Two-Point Separation: {'PASS' if test_passed else 'FAIL'}")
        
        return test_passed, self.test_results

if __name__ == "__main__":
    tester = ImprovedRangeResolutionTests()
    success, results = tester.run_improved_tests()
    
    # Save results
    import json
    with open('../output/improved_range_resolution_results.json', 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        json.dump(convert_numpy_types(results), f, indent=2)
    
    print(f"\nResults saved to: ../output/improved_range_resolution_results.json")
