#!/usr/bin/env python3
"""
Frequency Impact on SAR Resolution Study
Demonstrates how carrier frequency affects both range and azimuth resolution
Key relationships:
- Range Resolution: ΔR = c/(2*B) - Independent of frequency
- Azimuth Resolution: ΔAz = λ*R/(2*L_syn) = c*R/(2*fc*L_syn) - Inversely proportional to frequency
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

class FrequencyImpactResolutionStudy:
    def __init__(self):
        """Initialize frequency impact study"""
        self.c = 3e8  # Speed of light
        
        # Fixed system parameters (independent of frequency)
        self.B = 100e6  # Bandwidth (Hz) - affects range resolution
        self.Tp = 10e-6  # Pulse duration (s)
        self.platform_velocity = 200  # m/s
        self.integration_time = 1.0  # seconds
        self.target_range = 1000.0  # meters (reference range)
        
        # Calculate frequency-independent parameters
        self.synthetic_aperture_length = self.platform_velocity * self.integration_time
        self.range_resolution = self.c / (2 * self.B)  # Independent of frequency
        
        print(f"Frequency Impact on SAR Resolution Study")
        print(f"Fixed Parameters:")
        print(f"  Bandwidth: {self.B/1e6:.1f} MHz")
        print(f"  Range Resolution: {self.range_resolution:.3f} m (frequency independent)")
        print(f"  Platform Velocity: {self.platform_velocity} m/s")
        print(f"  Synthetic Aperture Length: {self.synthetic_aperture_length:.1f} m")
        print(f"  Reference Range: {self.target_range:.1f} m")
    
    def define_frequency_test_cases(self):
        """Define frequency test cases across different bands"""
        # SAR frequency bands with realistic values
        frequency_cases = {
            'P_band': {
                'frequency': 0.4e9,  # 400 MHz
                'band_name': 'P-band',
                'wavelength': None,  # Will be calculated
                'azimuth_resolution': None,  # Will be calculated
                'applications': 'Forest biomass, subsurface imaging'
            },
            'L_band': {
                'frequency': 1.25e9,  # 1.25 GHz
                'band_name': 'L-band', 
                'wavelength': None,
                'azimuth_resolution': None,
                'applications': 'ALOS PALSAR, soil moisture'
            },
            'S_band': {
                'frequency': 3.2e9,  # 3.2 GHz
                'band_name': 'S-band',
                'wavelength': None,
                'azimuth_resolution': None,
                'applications': 'Weather radar, some SAR systems'
            },
            'C_band': {
                'frequency': 5.4e9,  # 5.4 GHz
                'band_name': 'C-band',
                'wavelength': None,
                'azimuth_resolution': None,
                'applications': 'Sentinel-1, RADARSAT-2'
            },
            'X_band': {
                'frequency': 9.6e9,  # 9.6 GHz
                'band_name': 'X-band',
                'wavelength': None,
                'azimuth_resolution': None,
                'applications': 'TerraSAR-X, COSMO-SkyMed'
            },
            'Ku_band': {
                'frequency': 15e9,  # 15 GHz
                'band_name': 'Ku-band',
                'wavelength': None,
                'azimuth_resolution': None,
                'applications': 'High-resolution SAR'
            },
            'Ka_band': {
                'frequency': 35e9,  # 35 GHz
                'band_name': 'Ka-band',
                'wavelength': None,
                'azimuth_resolution': None,
                'applications': 'Very high-resolution SAR'
            }
        }
        
        # Calculate wavelength and azimuth resolution for each frequency
        for case_name, case_data in frequency_cases.items():
            fc = case_data['frequency']
            wavelength = self.c / fc
            azimuth_resolution = wavelength * self.target_range / (2 * self.synthetic_aperture_length)
            
            case_data['wavelength'] = wavelength
            case_data['azimuth_resolution'] = azimuth_resolution
        
        return frequency_cases
    
    def simulate_frequency_dependent_targets(self, fc, target_separations):
        """Simulate target responses for different frequencies"""
        print(f"\n  Testing frequency: {fc/1e9:.2f} GHz")
        
        # Initialize SAR model for this frequency
        sar = FinalSARModel(fc=fc, B=self.B, Tp=self.Tp)
        
        results = {}
        
        for sep_name, separation in target_separations.items():
            print(f"    {sep_name}: {separation:.3f}m separation")
            
            # Define two targets with this separation
            targets = [
                {'range': self.target_range, 'rcs': 1.0},
                {'range': self.target_range + separation, 'rcs': 1.0}
            ]
            
            # Generate individual responses
            t1, response1 = sar.point_target_response(R0=targets[0]['range'], fs=200e6, plot=False)
            t2, response2 = sar.point_target_response(R0=targets[1]['range'], fs=200e6, plot=False)
            
            # Combine responses
            min_len = min(len(response1), len(response2))
            combined_response = response1[:min_len] + response2[:min_len]
            
            # Range compression
            compressed = sar.range_compression(combined_response, plot=False)
            
            # Analysis
            magnitude = np.abs(compressed)
            range_axis = t1[:len(compressed)] * self.c / 2
            
            # Peak detection with adaptive thresholds
            best_peaks = []
            for threshold in [0.2, 0.3, 0.4, 0.5]:
                peaks_idx, _ = find_peaks(
                    magnitude,
                    height=threshold * np.max(magnitude),
                    distance=max(5, int(0.8 * self.range_resolution / (self.c/2) * 200e6))
                )
                
                if 1 <= len(peaks_idx) <= 3:
                    best_peaks = peaks_idx
                    break
            
            if len(best_peaks) == 0:
                best_peaks = [np.argmax(magnitude)]
            
            detected_ranges = range_axis[best_peaks]
            resolution_achieved = len(detected_ranges) >= 2
            
            results[sep_name] = {
                'separation': separation,
                'detected_peaks': len(detected_ranges),
                'detected_ranges': detected_ranges.tolist(),
                'resolution_achieved': resolution_achieved,
                'range_axis': range_axis,
                'magnitude': magnitude,
                'peak_indices': best_peaks
            }
            
            print(f"      Detected: {len(detected_ranges)} peaks, Resolved: {resolution_achieved}")
        
        return results
    
    def analyze_azimuth_resolution_vs_frequency(self, frequency_cases):
        """Analyze azimuth resolution across frequencies"""
        print(f"\nAzimuth Resolution Analysis:")
        print(f"Formula: ΔAz = λ*R/(2*L_syn) = c*R/(2*fc*L_syn)")
        print(f"Expected: Inversely proportional to frequency")
        
        azimuth_analysis = {}
        
        for case_name, case_data in frequency_cases.items():
            fc = case_data['frequency']
            wavelength = case_data['wavelength']
            azimuth_res = case_data['azimuth_resolution']
            
            # Calculate azimuth resolution for different target ranges
            ranges = [500, 1000, 2000, 5000]  # meters
            azimuth_vs_range = []
            
            for R in ranges:
                az_res = wavelength * R / (2 * self.synthetic_aperture_length)
                azimuth_vs_range.append(az_res)
            
            azimuth_analysis[case_name] = {
                'frequency': fc,
                'wavelength': wavelength,
                'azimuth_resolution_1km': azimuth_res,
                'azimuth_vs_range': {
                    'ranges': ranges,
                    'resolutions': azimuth_vs_range
                }
            }
            
            print(f"  {case_data['band_name']}: {fc/1e9:.2f} GHz, λ={wavelength*100:.1f}cm, ΔAz@1km={azimuth_res*100:.1f}cm")
        
        return azimuth_analysis
    
    def create_frequency_impact_plots(self, frequency_cases, range_results, azimuth_analysis):
        """Create comprehensive plots showing frequency impact"""
        fig = plt.figure(figsize=(20, 16))
        
        # Plot 1: Range Resolution vs Frequency (should be constant)
        ax1 = plt.subplot(2, 3, 1)
        frequencies = [case['frequency']/1e9 for case in frequency_cases.values()]
        range_resolutions = [self.range_resolution] * len(frequencies)  # Constant
        
        ax1.plot(frequencies, range_resolutions, 'bo-', linewidth=3, markersize=10)
        ax1.set_xlabel('Frequency (GHz)', fontsize=12)
        ax1.set_ylabel('Range Resolution (m)', fontsize=12)
        ax1.set_title('Range Resolution vs Frequency\n(Independent of Frequency)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 2*self.range_resolution)
        
        # Add annotation
        ax1.text(0.5, 0.8, f'ΔR = c/(2B) = {self.range_resolution:.3f} m\n(Constant)', 
                transform=ax1.transAxes, fontsize=12, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Plot 2: Azimuth Resolution vs Frequency
        ax2 = plt.subplot(2, 3, 2)
        azimuth_resolutions = [case['azimuth_resolution']*100 for case in frequency_cases.values()]  # Convert to cm
        
        ax2.plot(frequencies, azimuth_resolutions, 'ro-', linewidth=3, markersize=10)
        ax2.set_xlabel('Frequency (GHz)', fontsize=12)
        ax2.set_ylabel('Azimuth Resolution (cm)', fontsize=12)
        ax2.set_title('Azimuth Resolution vs Frequency\n(Inversely Proportional)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # Add theoretical curve
        freq_theory = np.linspace(min(frequencies), max(frequencies), 100)
        az_theory = (self.c * self.target_range / (2 * freq_theory * 1e9 * self.synthetic_aperture_length)) * 100
        ax2.plot(freq_theory, az_theory, 'r--', alpha=0.7, linewidth=2, label='Theory: ΔAz ∝ 1/fc')
        ax2.legend()
        
        # Plot 3: Wavelength vs Frequency
        ax3 = plt.subplot(2, 3, 3)
        wavelengths = [case['wavelength']*100 for case in frequency_cases.values()]  # Convert to cm
        
        ax3.plot(frequencies, wavelengths, 'go-', linewidth=3, markersize=10)
        ax3.set_xlabel('Frequency (GHz)', fontsize=12)
        ax3.set_ylabel('Wavelength (cm)', fontsize=12)
        ax3.set_title('Wavelength vs Frequency\n(λ = c/f)', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        
        # Add theoretical curve
        wavelength_theory = (self.c / (freq_theory * 1e9)) * 100
        ax3.plot(freq_theory, wavelength_theory, 'g--', alpha=0.7, linewidth=2, label='Theory: λ = c/f')
        ax3.legend()
        
        # Plot 4: Azimuth Resolution vs Range for Different Frequencies
        ax4 = plt.subplot(2, 3, 4)
        ranges = [500, 1000, 2000, 5000]
        colors = ['purple', 'blue', 'green', 'orange', 'red', 'brown', 'pink']
        
        for i, (case_name, analysis) in enumerate(azimuth_analysis.items()):
            if i < len(colors):
                fc = analysis['frequency']
                resolutions = [r*100 for r in analysis['azimuth_vs_range']['resolutions']]  # Convert to cm
                ax4.plot(ranges, resolutions, 'o-', color=colors[i], linewidth=2, 
                        label=f'{fc/1e9:.1f} GHz', markersize=8)
        
        ax4.set_xlabel('Range (m)', fontsize=12)
        ax4.set_ylabel('Azimuth Resolution (cm)', fontsize=12)
        ax4.set_title('Azimuth Resolution vs Range\n(Different Frequencies)', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=10)
        ax4.set_yscale('log')
        ax4.set_xscale('log')
        
        # Plot 5: SAR Band Comparison
        ax5 = plt.subplot(2, 3, 5)
        band_names = [case['band_name'] for case in frequency_cases.values()]
        azimuth_res_cm = [case['azimuth_resolution']*100 for case in frequency_cases.values()]
        
        bars = ax5.bar(range(len(band_names)), azimuth_res_cm, 
                      color=['purple', 'blue', 'green', 'orange', 'red', 'brown', 'pink'][:len(band_names)])
        ax5.set_xlabel('SAR Band', fontsize=12)
        ax5.set_ylabel('Azimuth Resolution (cm)', fontsize=12)
        ax5.set_title('SAR Band Comparison\n(Azimuth Resolution at 1km)', fontsize=14, fontweight='bold')
        ax5.set_xticks(range(len(band_names)))
        ax5.set_xticklabels(band_names, rotation=45)
        ax5.grid(True, alpha=0.3, axis='y')
        ax5.set_yscale('log')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, azimuth_res_cm)):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                    f'{value:.1f}cm', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Plot 6: Resolution Comparison Summary
        ax6 = plt.subplot(2, 3, 6)
        
        # Create summary table
        summary_data = []
        for case_name, case_data in frequency_cases.items():
            freq_ghz = case_data['frequency'] / 1e9
            wavelength_cm = case_data['wavelength'] * 100
            azimuth_cm = case_data['azimuth_resolution'] * 100
            range_m = self.range_resolution
            
            summary_data.append([
                case_data['band_name'],
                f'{freq_ghz:.1f}',
                f'{wavelength_cm:.1f}',
                f'{range_m:.3f}',
                f'{azimuth_cm:.1f}'
            ])
        
        # Create table
        table = ax6.table(cellText=summary_data,
                         colLabels=['Band', 'Freq\n(GHz)', 'λ\n(cm)', 'Range Res\n(m)', 'Azimuth Res\n(cm)'],
                         cellLoc='center',
                         loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(summary_data) + 1):
            for j in range(5):
                cell = table[(i, j)]
                if i == 0:  # Header row
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        ax6.set_title('SAR Frequency Band Summary', fontsize=14, fontweight='bold')
        ax6.axis('off')
        
        plt.suptitle('Frequency Impact on SAR Resolution Analysis', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig('../output/frequency_impact_resolution_study.png', dpi=300, bbox_inches='tight')
        print("\nFrequency impact resolution study plots saved to: ../output/frequency_impact_resolution_study.png")
        plt.close()
    
    def run_frequency_impact_study(self):
        """Run comprehensive frequency impact study"""
        print("FREQUENCY IMPACT ON SAR RESOLUTION STUDY")
        print("="*80)
        
        # Create output directory
        os.makedirs('../output', exist_ok=True)
        
        # Define frequency test cases
        frequency_cases = self.define_frequency_test_cases()
        
        # Define target separations to test range resolution
        target_separations = {
            'well_separated': 5.0,    # 5m - should resolve at all frequencies
            'marginal': 2.0,          # 2m - marginal resolution
            'close': 1.0              # 1m - below range resolution limit
        }
        
        print(f"\nRange Resolution Analysis:")
        print(f"Testing separations: {list(target_separations.values())} meters")
        print(f"Range resolution (constant): {self.range_resolution:.3f} m")
        
        # Test range resolution across frequencies
        range_results = {}
        for case_name, case_data in frequency_cases.items():
            fc = case_data['frequency']
            range_results[case_name] = self.simulate_frequency_dependent_targets(fc, target_separations)
        
        # Analyze azimuth resolution
        azimuth_analysis = self.analyze_azimuth_resolution_vs_frequency(frequency_cases)
        
        # Create comprehensive plots
        self.create_frequency_impact_plots(frequency_cases, range_results, azimuth_analysis)
        
        # Summary analysis
        print(f"\n" + "="*80)
        print("FREQUENCY IMPACT SUMMARY")
        print("="*80)
        
        print(f"\n1. RANGE RESOLUTION:")
        print(f"   - Formula: ΔR = c/(2*B)")
        print(f"   - Value: {self.range_resolution:.3f} m (CONSTANT across all frequencies)")
        print(f"   - Depends on: Bandwidth (B), NOT frequency (fc)")
        
        print(f"\n2. AZIMUTH RESOLUTION:")
        print(f"   - Formula: ΔAz = λ*R/(2*L_syn) = c*R/(2*fc*L_syn)")
        print(f"   - Relationship: INVERSELY proportional to frequency")
        print(f"   - At 1km range:")
        
        for case_name, case_data in frequency_cases.items():
            freq_ghz = case_data['frequency'] / 1e9
            azimuth_cm = case_data['azimuth_resolution'] * 100
            improvement_factor = frequency_cases['P_band']['azimuth_resolution'] / case_data['azimuth_resolution']
            print(f"     {case_data['band_name']:8}: {freq_ghz:5.1f} GHz → {azimuth_cm:6.1f} cm ({improvement_factor:4.1f}x better than P-band)")
        
        print(f"\n3. KEY INSIGHTS:")
        print(f"   - Higher frequency → Better azimuth resolution")
        print(f"   - Higher frequency → Same range resolution")
        print(f"   - Ka-band (35 GHz) has {frequency_cases['P_band']['azimuth_resolution']/frequency_cases['Ka_band']['azimuth_resolution']:.1f}x better azimuth resolution than P-band (0.4 GHz)")
        print(f"   - Range resolution only improves with increased bandwidth (B)")
        
        # Save detailed results
        study_results = {
            'timestamp': datetime.now().isoformat(),
            'study_type': 'frequency_impact_resolution',
            'fixed_parameters': {
                'bandwidth': self.B,
                'range_resolution': self.range_resolution,
                'platform_velocity': self.platform_velocity,
                'synthetic_aperture_length': self.synthetic_aperture_length,
                'reference_range': self.target_range
            },
            'frequency_cases': {
                case_name: {
                    'frequency': case_data['frequency'],
                    'band_name': case_data['band_name'],
                    'wavelength': case_data['wavelength'],
                    'azimuth_resolution_1km': case_data['azimuth_resolution'],
                    'applications': case_data['applications']
                }
                for case_name, case_data in frequency_cases.items()
            },
            'key_relationships': {
                'range_resolution': 'ΔR = c/(2*B) - Independent of frequency',
                'azimuth_resolution': 'ΔAz = λ*R/(2*L_syn) = c*R/(2*fc*L_syn) - Inversely proportional to frequency'
            }
        }
        
        with open('../output/frequency_impact_study_results.json', 'w') as f:
            json.dump(study_results, f, indent=2)
        
        print(f"\nDetailed results saved to: ../output/frequency_impact_study_results.json")
        
        return frequency_cases, range_results, azimuth_analysis

def main():
    """Main execution"""
    study = FrequencyImpactResolutionStudy()
    frequency_cases, range_results, azimuth_analysis = study.run_frequency_impact_study()
    return study, frequency_cases, range_results, azimuth_analysis

if __name__ == "__main__":
    study, frequency_cases, range_results, azimuth_analysis = main()
