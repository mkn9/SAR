#!/usr/bin/env python3
"""
Azimuth-Corrected SAR Target Resolution Experiment
Demonstrates both range and azimuth resolution capabilities with proper 2D processing
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

class AzimuthCorrectedResolutionExperiment:
    def __init__(self, fc=10e9, B=100e6, Tp=10e-6, PRF=1000, platform_velocity=200):
        """Initialize azimuth-corrected SAR experiment"""
        self.fc = fc
        self.B = B
        self.Tp = Tp
        self.PRF = PRF
        self.platform_velocity = platform_velocity
        self.c = 3e8
        
        # Initialize SAR model
        self.sar = FinalSARModel(fc=fc, B=B, Tp=Tp)
        
        # Calculate system parameters
        self.wavelength = self.c / fc
        self.range_resolution = self.c / (2 * B)
        self.fs = 200e6  # SAR model's sampling frequency
        
        # Azimuth processing parameters
        self.integration_time = 1.0  # seconds
        self.synthetic_aperture_length = platform_velocity * self.integration_time
        self.azimuth_resolution = self.wavelength * 1000 / (2 * self.synthetic_aperture_length)  # At 1km range
        
        print(f"Azimuth-Corrected SAR Experiment Initialized:")
        print(f"Range Resolution: {self.range_resolution:.3f} m")
        print(f"Azimuth Resolution (at 1km): {self.azimuth_resolution:.3f} m")
        print(f"Synthetic Aperture Length: {self.synthetic_aperture_length:.1f} m")
        print(f"Platform Velocity: {self.platform_velocity} m/s")
    
    def define_comprehensive_scenarios(self):
        """Define comprehensive test scenarios for both range and azimuth resolution"""
        scenarios = {
            'range_resolved': {
                'description': 'Two targets separated in range - should resolve',
                'targets': [
                    {'range': 1000.0, 'azimuth': 0.0, 'rcs': 1.0, 'name': 'Target_A'},
                    {'range': 1003.0, 'azimuth': 0.0, 'rcs': 1.0, 'name': 'Target_B'}  # 3m separation > 1.5m resolution
                ],
                'separation_type': 'range',
                'separation_distance': 3.0,
                'expected_resolution': True
            },
            'range_unresolved': {
                'description': 'Two targets too close in range - should not resolve',
                'targets': [
                    {'range': 1000.0, 'azimuth': 0.0, 'rcs': 1.0, 'name': 'Target_C'},
                    {'range': 1000.8, 'azimuth': 0.0, 'rcs': 1.0, 'name': 'Target_D'}  # 0.8m < 1.5m resolution
                ],
                'separation_type': 'range',
                'separation_distance': 0.8,
                'expected_resolution': False
            },
            'azimuth_resolved': {
                'description': 'Two targets separated in azimuth - should resolve',
                'targets': [
                    {'range': 1000.0, 'azimuth': -0.15, 'rcs': 1.0, 'name': 'Target_E'},  # -0.15m azimuth
                    {'range': 1000.0, 'azimuth': 0.15, 'rcs': 1.0, 'name': 'Target_F'}   # +0.15m azimuth
                ],
                'separation_type': 'azimuth',
                'separation_distance': 0.3,  # 0.3m > 0.075m azimuth resolution
                'expected_resolution': True
            },
            'azimuth_unresolved': {
                'description': 'Two targets too close in azimuth - should not resolve',
                'targets': [
                    {'range': 1000.0, 'azimuth': -0.03, 'rcs': 1.0, 'name': 'Target_G'},  # -0.03m azimuth
                    {'range': 1000.0, 'azimuth': 0.03, 'rcs': 1.0, 'name': 'Target_H'}   # +0.03m azimuth
                ],
                'separation_type': 'azimuth',
                'separation_distance': 0.06,  # 0.06m < 0.075m azimuth resolution
                'expected_resolution': False
            }
        }
        return scenarios
    
    def simulate_azimuth_sar_data(self, targets, num_azimuth_positions=64):
        """Simulate 2D SAR data with azimuth processing"""
        print(f"  Simulating 2D SAR data for {len(targets)} targets:")
        for target in targets:
            print(f"    {target['name']}: Range={target['range']:.1f}m, Azimuth={target.get('azimuth', 0):.3f}m")
        
        # Define azimuth positions (platform positions)
        azimuth_extent = self.synthetic_aperture_length / 2
        azimuth_positions = np.linspace(-azimuth_extent, azimuth_extent, num_azimuth_positions)
        
        # Generate SAR data matrix
        # First, get the time vector from a reference target
        t_ref, _ = self.sar.point_target_response(R0=1000.0, fs=self.fs, plot=False)
        num_range_samples = len(t_ref)
        
        # Initialize 2D data matrix [range_samples, azimuth_samples]
        sar_data = np.zeros((num_range_samples, num_azimuth_positions), dtype=complex)
        
        # For each azimuth position, simulate the received signal
        for az_idx, platform_azimuth in enumerate(azimuth_positions):
            combined_signal = np.zeros(num_range_samples, dtype=complex)
            
            for target in targets:
                target_range = target['range']
                target_azimuth = target.get('azimuth', 0.0)
                target_rcs = target['rcs']
                
                # Calculate range from platform to target
                # Simplified geometry: R = sqrt(R0^2 + (platform_azimuth - target_azimuth)^2)
                azimuth_diff = platform_azimuth - target_azimuth
                instantaneous_range = np.sqrt(target_range**2 + azimuth_diff**2)
                
                # Generate point target response at this range
                t, target_response = self.sar.point_target_response(
                    R0=instantaneous_range, fs=self.fs, plot=False
                )
                
                # Ensure same length
                min_len = min(len(combined_signal), len(target_response))
                combined_signal[:min_len] += target_response[:min_len] * target_rcs
            
            # Store in SAR data matrix
            sar_data[:, az_idx] = combined_signal
        
        return sar_data, t_ref, azimuth_positions
    
    def process_2d_sar_data(self, sar_data):
        """Process 2D SAR data with range and azimuth compression"""
        print("  Processing 2D SAR data...")
        
        # Range compression for each azimuth position
        range_compressed = np.zeros_like(sar_data)
        for az_idx in range(sar_data.shape[1]):
            range_compressed[:, az_idx] = self.sar.range_compression(sar_data[:, az_idx], plot=False)
        
        # Simple azimuth compression (coherent integration)
        # In a full SAR processor, this would include range migration correction
        azimuth_compressed = np.zeros_like(range_compressed)
        for range_idx in range(range_compressed.shape[0]):
            # Apply window function to reduce sidelobes
            azimuth_line = range_compressed[range_idx, :]
            window = np.hanning(len(azimuth_line))
            windowed_line = azimuth_line * window
            
            # FFT-based azimuth compression (simplified)
            azimuth_fft = np.fft.fft(windowed_line)
            azimuth_compressed[range_idx, :] = np.fft.ifft(azimuth_fft)
        
        return azimuth_compressed
    
    def analyze_2d_resolution(self, processed_image, time_vector, azimuth_positions, targets, scenario_name):
        """Analyze 2D resolution performance"""
        # Calculate axes
        range_axis = time_vector * self.c / 2
        azimuth_axis = azimuth_positions
        
        # Get magnitude image
        magnitude_image = np.abs(processed_image)
        
        # Find peaks in the 2D image
        # Method 1: Sum across azimuth to get range profile
        range_profile = np.sum(magnitude_image, axis=1)
        # Method 2: Sum across range to get azimuth profile  
        azimuth_profile = np.sum(magnitude_image, axis=0)
        
        # Peak detection in range
        range_peaks_idx, _ = find_peaks(
            range_profile,
            height=0.3 * np.max(range_profile),
            distance=int(self.range_resolution / (self.c/2) * self.fs)
        )
        detected_ranges = range_axis[range_peaks_idx] if len(range_peaks_idx) > 0 else []
        
        # Peak detection in azimuth
        azimuth_sample_spacing = azimuth_positions[1] - azimuth_positions[0]
        azimuth_distance_samples = max(1, int(self.azimuth_resolution / azimuth_sample_spacing))
        
        azimuth_peaks_idx, _ = find_peaks(
            azimuth_profile,
            height=0.3 * np.max(azimuth_profile),
            distance=azimuth_distance_samples
        )
        detected_azimuths = azimuth_axis[azimuth_peaks_idx] if len(azimuth_peaks_idx) > 0 else []
        
        # Determine resolution achievement
        expected_ranges = [target['range'] for target in targets]
        expected_azimuths = [target.get('azimuth', 0.0) for target in targets]
        
        # For range-separated targets, check range resolution
        # For azimuth-separated targets, check azimuth resolution
        if scenario_name in ['range_resolved', 'range_unresolved']:
            resolution_achieved = len(detected_ranges) >= len(targets)
            primary_detections = detected_ranges
            expected_primary = expected_ranges
        else:  # azimuth scenarios
            resolution_achieved = len(detected_azimuths) >= len(targets)
            primary_detections = detected_azimuths
            expected_primary = expected_azimuths
        
        return {
            'scenario': scenario_name,
            'num_targets': len(targets),
            'expected_ranges': expected_ranges,
            'expected_azimuths': expected_azimuths,
            'detected_ranges': detected_ranges,
            'detected_azimuths': detected_azimuths,
            'resolution_achieved': resolution_achieved,
            'range_profile': range_profile,
            'azimuth_profile': azimuth_profile,
            'range_axis': range_axis,
            'azimuth_axis': azimuth_axis,
            'magnitude_image': magnitude_image
        }
    
    def create_comprehensive_plots(self, results):
        """Create comprehensive plots showing both range and azimuth resolution"""
        fig = plt.figure(figsize=(20, 16))
        
        # Create a 3x4 subplot layout
        scenarios = list(results.keys())[:4]
        
        for idx, scenario_name in enumerate(scenarios):
            result = results[scenario_name]
            
            # 2D SAR Image
            ax1 = plt.subplot(3, 4, idx + 1)
            magnitude_image = result['magnitude_image']
            range_axis = result['range_axis']
            azimuth_axis = result['azimuth_axis']
            
            # Display 2D image
            extent = [azimuth_axis[0], azimuth_axis[-1], range_axis[-1], range_axis[0]]
            im = ax1.imshow(magnitude_image, aspect='auto', extent=extent, cmap='jet')
            
            # Mark expected targets
            expected_ranges = result['expected_ranges']
            expected_azimuths = result['expected_azimuths']
            for i, (r, az) in enumerate(zip(expected_ranges, expected_azimuths)):
                ax1.plot(az, r, 'wo', markersize=8, markeredgecolor='black', linewidth=2,
                        label=f'Target {i+1}' if i < 2 else '')
            
            resolution_status = "RESOLVED" if result['resolution_achieved'] else "UNRESOLVED"
            color = 'green' if result['resolution_achieved'] else 'red'
            ax1.set_title(f"{scenario_name}\n{resolution_status}", color=color, fontweight='bold')
            ax1.set_xlabel('Azimuth (m)')
            ax1.set_ylabel('Range (m)')
            if idx == 0:
                ax1.legend()
            
            # Range Profile
            ax2 = plt.subplot(3, 4, idx + 5)
            range_profile = result['range_profile']
            range_axis = result['range_axis']
            
            ax2.plot(range_axis, range_profile, 'b-', linewidth=1.5, label='Range Profile')
            
            # Mark expected and detected ranges
            for i, exp_range in enumerate(expected_ranges):
                ax2.axvline(exp_range, color='red', linestyle='--', alpha=0.8, linewidth=2,
                           label=f'Expected {i+1}' if i < 2 and idx == 0 else '')
            
            detected_ranges = result['detected_ranges']
            for i, det_range in enumerate(detected_ranges):
                ax2.axvline(det_range, color='green', linestyle='-', alpha=0.9, linewidth=2,
                           label=f'Detected {i+1}' if i < 2 and idx == 0 else '')
            
            ax2.set_xlabel('Range (m)')
            ax2.set_ylabel('Magnitude')
            ax2.set_title('Range Profile')
            if idx == 0:
                ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Azimuth Profile
            ax3 = plt.subplot(3, 4, idx + 9)
            azimuth_profile = result['azimuth_profile']
            azimuth_axis = result['azimuth_axis']
            
            ax3.plot(azimuth_axis, azimuth_profile, 'g-', linewidth=1.5, label='Azimuth Profile')
            
            # Mark expected and detected azimuths
            for i, exp_azimuth in enumerate(expected_azimuths):
                ax3.axvline(exp_azimuth, color='red', linestyle='--', alpha=0.8, linewidth=2,
                           label=f'Expected {i+1}' if i < 2 and idx == 0 else '')
            
            detected_azimuths = result['detected_azimuths']
            for i, det_azimuth in enumerate(detected_azimuths):
                ax3.axvline(det_azimuth, color='orange', linestyle='-', alpha=0.9, linewidth=2,
                           label=f'Detected {i+1}' if i < 2 and idx == 0 else '')
            
            ax3.set_xlabel('Azimuth (m)')
            ax3.set_ylabel('Magnitude')
            ax3.set_title('Azimuth Profile')
            if idx == 0:
                ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        plt.suptitle('Azimuth-Corrected SAR Target Resolution Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('../output/azimuth_corrected_resolution_plots.png', dpi=300, bbox_inches='tight')
        print("Azimuth-corrected resolution plots saved to: ../output/azimuth_corrected_resolution_plots.png")
        plt.close()
    
    def run_experiment(self):
        """Run the comprehensive azimuth-corrected experiment"""
        print("AZIMUTH-CORRECTED SAR TARGET RESOLUTION EXPERIMENT")
        print("="*70)
        
        # Create output directory
        os.makedirs('../output', exist_ok=True)
        
        # Define scenarios
        scenarios = self.define_comprehensive_scenarios()
        results = {}
        
        for scenario_name, scenario in scenarios.items():
            print(f"\n--- Processing: {scenario_name} ---")
            print(f"Description: {scenario['description']}")
            print(f"Separation: {scenario['separation_distance']:.3f}m")
            print(f"Type: {scenario['separation_type']}")
            
            if scenario['separation_type'] == 'range':
                print(f"vs Range Resolution: {self.range_resolution:.3f}m")
            else:
                print(f"vs Azimuth Resolution: {self.azimuth_resolution:.3f}m")
            
            # Simulate 2D SAR data
            sar_data, time_vector, azimuth_positions = self.simulate_azimuth_sar_data(scenario['targets'])
            
            # Process 2D SAR data
            processed_image = self.process_2d_sar_data(sar_data)
            
            # Analyze resolution
            analysis = self.analyze_2d_resolution(
                processed_image, time_vector, azimuth_positions, scenario['targets'], scenario_name
            )
            
            results[scenario_name] = analysis
            
            # Print results
            expected = scenario['expected_resolution']
            achieved = analysis['resolution_achieved']
            status = "✓" if expected == achieved else "✗"
            
            print(f"  Expected Resolution: {expected}")
            print(f"  Achieved Resolution: {achieved} {status}")
            print(f"  Targets: {analysis['num_targets']}")
            print(f"  Range Detections: {len(analysis['detected_ranges'])}")
            print(f"  Azimuth Detections: {len(analysis['detected_azimuths'])}")
        
        # Create comprehensive plots
        self.create_comprehensive_plots(results)
        
        # Summary
        print(f"\n" + "="*70)
        print("EXPERIMENT SUMMARY")
        print("="*70)
        
        correct_predictions = 0
        total_tests = len(scenarios)
        
        for scenario_name, scenario in scenarios.items():
            result = results[scenario_name]
            expected = scenario['expected_resolution']
            achieved = result['resolution_achieved']
            status = "✓" if expected == achieved else "✗"
            sep_type = scenario['separation_type']
            sep_dist = scenario['separation_distance']
            print(f"{scenario_name}: {sep_type.upper()} sep={sep_dist:.3f}m, Expected {expected}, Got {achieved} {status}")
            if expected == achieved:
                correct_predictions += 1
        
        success_rate = correct_predictions / total_tests * 100
        print(f"\nOverall Success Rate: {correct_predictions}/{total_tests} ({success_rate:.1f}%)")
        
        return results

def main():
    """Main execution"""
    experiment = AzimuthCorrectedResolutionExperiment()
    results = experiment.run_experiment()
    return experiment, results

if __name__ == "__main__":
    experiment, results = main()
