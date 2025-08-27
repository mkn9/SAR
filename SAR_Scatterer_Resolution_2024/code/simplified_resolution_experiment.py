#!/usr/bin/env python3
"""
Simplified SAR Scatterer Resolution Experiment
Focus on correct range detection without complex 2D processing
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append('../../SAR_Benchmark_Validation_2024/code')
from sar_model_final import FinalSARModel
from scipy.signal import find_peaks
import json
from datetime import datetime

class SimplifiedScattererExperiment:
    def __init__(self, fc=10e9, B=100e6, Tp=10e-6):
        """Initialize simplified experiment"""
        self.fc = fc
        self.B = B
        self.Tp = Tp
        self.c = 3e8
        
        # Initialize SAR model
        self.sar = FinalSARModel(fc=fc, B=B, Tp=Tp)
        
        # Calculate system parameters
        self.wavelength = self.c / fc
        self.range_resolution = self.c / (2 * B)
        self.fs = 200e6  # Use SAR model's default sampling frequency
        
        print(f"Simplified SAR Experiment Initialized:")
        print(f"Range Resolution: {self.range_resolution:.3f} m")
        print(f"Sampling Frequency: {self.fs/1e6:.1f} MHz")
    
    def define_test_scenarios(self):
        """Define test scenarios with proper separation analysis"""
        scenarios = {
            'range_resolved': {
                'description': 'Two targets separated in range - should resolve',
                'targets': [
                    {'range': 1000.0, 'rcs': 1.0, 'name': 'Target_A'},
                    {'range': 1002.0, 'rcs': 1.0, 'name': 'Target_B'}
                ],
                'separation_distance': 2.0,
                'expected_resolution': True
            },
            'range_unresolved': {
                'description': 'Two targets too close in range - should not resolve',
                'targets': [
                    {'range': 1000.0, 'rcs': 1.0, 'name': 'Target_C'},
                    {'range': 1000.8, 'rcs': 1.0, 'name': 'Target_D'}
                ],
                'separation_distance': 0.8,
                'expected_resolution': False
            },
            'well_resolved': {
                'description': 'Two targets well separated - should clearly resolve',
                'targets': [
                    {'range': 1000.0, 'rcs': 1.0, 'name': 'Target_E'},
                    {'range': 1005.0, 'rcs': 1.0, 'name': 'Target_F'}
                ],
                'separation_distance': 5.0,
                'expected_resolution': True
            }
        }
        return scenarios
    
    def simulate_multi_target_response(self, targets):
        """Simulate multi-target response using simple superposition"""
        print(f"  Simulating {len(targets)} targets:")
        for target in targets:
            print(f"    {target['name']}: {target['range']}m, RCS={target['rcs']}")
        
        # Generate individual responses
        responses = []
        time_vectors = []
        
        for target in targets:
            t, response = self.sar.point_target_response(
                R0=target['range'], fs=self.fs, plot=False
            )
            responses.append(response * target['rcs'])
            time_vectors.append(t)
        
        # Find common time vector (use the longest one)
        max_len = max(len(t) for t in time_vectors)
        common_time = time_vectors[0][:max_len] if len(time_vectors[0]) >= max_len else time_vectors[0]
        
        # Pad all responses to same length and sum
        combined_response = np.zeros(len(common_time), dtype=complex)
        for response in responses:
            min_len = min(len(combined_response), len(response))
            combined_response[:min_len] += response[:min_len]
        
        return common_time, combined_response
    
    def analyze_target_resolution(self, time_vector, combined_response, targets, scenario_name):
        """Analyze resolution performance"""
        # Range compress the combined signal
        compressed = self.sar.range_compression(combined_response, plot=False)
        
        # Ensure same length
        min_len = min(len(time_vector), len(compressed))
        time_vector = time_vector[:min_len]
        compressed = compressed[:min_len]
        
        # Calculate range axis and magnitude
        range_axis = time_vector * self.c / 2
        magnitude = np.abs(compressed)
        
        # Improved peak detection
        resolution_distance_samples = int(self.range_resolution / (self.c/2) * self.fs)
        
        # Try different thresholds
        best_peaks = []
        for threshold in [0.2, 0.3, 0.5, 0.7]:
            peaks_idx, properties = find_peaks(
                magnitude,
                height=threshold * np.max(magnitude),
                distance=max(resolution_distance_samples, 10),
                prominence=0.05 * np.max(magnitude)
            )
            
            # Select reasonable number of peaks
            if 1 <= len(peaks_idx) <= len(targets) + 2:
                best_peaks = peaks_idx
                break
        
        if len(best_peaks) == 0:
            # Fallback: find global maximum
            best_peaks = [np.argmax(magnitude)]
        
        # Convert to ranges
        detected_ranges = range_axis[best_peaks]
        peak_magnitudes = magnitude[best_peaks]
        expected_ranges = [target['range'] for target in targets]
        
        # Calculate resolution performance
        resolution_achieved = len(detected_ranges) >= len(targets)
        
        return {
            'scenario': scenario_name,
            'num_targets': len(targets),
            'expected_ranges': expected_ranges,
            'detected_peaks': len(detected_ranges),
            'detected_ranges': detected_ranges.tolist(),
            'peak_magnitudes': peak_magnitudes.tolist(),
            'resolution_achieved': resolution_achieved,
            'range_axis': range_axis,
            'magnitude': magnitude,
            'time_vector': time_vector
        }
    
    def create_resolution_plots(self, results):
        """Create corrected resolution plots"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Corrected SAR Target Resolution Analysis', fontsize=16)
        
        scenario_items = list(results.items())[:4]  # Take first 4 scenarios
        
        for idx, (scenario_name, result) in enumerate(scenario_items):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            # Plot range profile
            range_axis = result['range_axis']
            magnitude = result['magnitude']
            
            ax.plot(range_axis, magnitude, 'b-', linewidth=1, label='SAR Response')
            
            # Mark expected targets
            expected_ranges = result['expected_ranges']
            colors_expected = ['red', 'orange', 'purple', 'brown']
            for i, target_range in enumerate(expected_ranges):
                color = colors_expected[i] if i < len(colors_expected) else 'red'
                ax.axvline(target_range, color=color, linestyle='--', alpha=0.8, linewidth=2,
                          label=f'Expected {i+1}' if i < 2 else '')
            
            # Mark detected peaks
            detected_ranges = result['detected_ranges']
            peak_magnitudes = result['peak_magnitudes']
            colors_detected = ['green', 'cyan', 'magenta', 'yellow']
            
            for i, (det_range, magnitude_val) in enumerate(zip(detected_ranges, peak_magnitudes)):
                color = colors_detected[i] if i < len(colors_detected) else 'green'
                ax.plot(det_range, magnitude_val, 'o', color=color, markersize=10,
                       label=f'Detected {i+1}' if i < 2 else '')
            
            # Formatting
            resolution_status = "RESOLVED" if result['resolution_achieved'] else "UNRESOLVED"
            status_color = 'green' if result['resolution_achieved'] else 'red'
            
            ax.set_title(f"{scenario_name}: {resolution_status}\n"
                        f"Expected: {result['num_targets']}, "
                        f"Detected: {result['detected_peaks']}", 
                        color=status_color, fontweight='bold')
            ax.set_xlabel('Range (m)')
            ax.set_ylabel('Magnitude')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Set reasonable range limits
            if expected_ranges:
                center_range = np.mean(expected_ranges)
                range_span = max(10, max(expected_ranges) - min(expected_ranges) + 10)
                ax.set_xlim(center_range - range_span, center_range + range_span)
        
        plt.tight_layout()
        plt.savefig('../output/corrected_resolution_plots.png', dpi=300, bbox_inches='tight')
        print("Corrected resolution plots saved to: ../output/corrected_resolution_plots.png")
        plt.close()
    
    def run_experiment(self):
        """Run the simplified experiment"""
        print("SIMPLIFIED SAR SCATTERER RESOLUTION EXPERIMENT")
        print("="*60)
        
        # Create output directory
        os.makedirs('../output', exist_ok=True)
        
        # Define scenarios
        scenarios = self.define_test_scenarios()
        results = {}
        
        for scenario_name, scenario in scenarios.items():
            print(f"\n--- Processing: {scenario_name} ---")
            print(f"Description: {scenario['description']}")
            print(f"Separation: {scenario['separation_distance']:.1f}m vs Resolution: {self.range_resolution:.1f}m")
            
            # Simulate multi-target response
            time_vector, combined_response = self.simulate_multi_target_response(scenario['targets'])
            
            # Analyze resolution
            analysis = self.analyze_target_resolution(
                time_vector, combined_response, scenario['targets'], scenario_name
            )
            
            results[scenario_name] = analysis
            
            # Print results
            expected = scenario['expected_resolution']
            achieved = analysis['resolution_achieved']
            status = "✓" if expected == achieved else "✗"
            
            print(f"  Expected Resolution: {expected}")
            print(f"  Achieved Resolution: {achieved} {status}")
            print(f"  Targets: {analysis['num_targets']}, Detected: {analysis['detected_peaks']}")
            print(f"  Expected Ranges: {[f'{r:.1f}' for r in analysis['expected_ranges']]}")
            print(f"  Detected Ranges: {[f'{r:.1f}' for r in analysis['detected_ranges']]}")
            
            # Calculate range errors
            if analysis['detected_ranges']:
                expected_ranges = analysis['expected_ranges']
                detected_ranges = analysis['detected_ranges']
                
                errors = []
                for exp_range in expected_ranges:
                    closest_det = min(detected_ranges, key=lambda x: abs(x - exp_range))
                    error = abs(closest_det - exp_range)
                    errors.append(error)
                
                avg_error = np.mean(errors)
                print(f"  Average Range Error: {avg_error:.1f} m")
        
        # Create corrected plots
        self.create_resolution_plots(results)
        
        # Summary
        print(f"\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)
        
        correct_predictions = 0
        total_tests = len(scenarios)
        
        for scenario_name, scenario in scenarios.items():
            result = results[scenario_name]
            expected = scenario['expected_resolution']
            achieved = result['resolution_achieved']
            status = "✓" if expected == achieved else "✗"
            print(f"{scenario_name}: Expected {expected}, Got {achieved} {status}")
            if expected == achieved:
                correct_predictions += 1
        
        success_rate = correct_predictions / total_tests * 100
        print(f"\nOverall Success Rate: {correct_predictions}/{total_tests} ({success_rate:.1f}%)")
        
        return results

def main():
    """Main execution"""
    experiment = SimplifiedScattererExperiment()
    results = experiment.run_experiment()
    return experiment, results

if __name__ == "__main__":
    experiment, results = main()
