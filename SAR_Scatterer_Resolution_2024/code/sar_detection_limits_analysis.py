#!/usr/bin/env python3
"""
SAR Detection Limits Analysis
Comprehensive study of target detectability based on radar equation parameters:

Key Parameters Analyzed:
- Transmit Power (Pt)
- Target Range (R)
- Target Radar Cross Section (RCS)
- System Noise Figure
- Integration Time
- Antenna Gain

Radar Equation: Pr = (Pt * G^2 * λ^2 * σ) / ((4π)^3 * R^4 * L)
SNR = Pr / (k * T * B * F * L)

Detection Criteria:
- Minimum SNR for detection
- False alarm and detection probabilities
- Coherent vs non-coherent integration
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append('../../SAR_Benchmark_Validation_2024/code')
from sar_model_final import FinalSARModel
from scipy import stats
import json
from datetime import datetime

class SARDetectionLimitsAnalysis:
    def __init__(self, fc=10e9, B=100e6, antenna_diameter=3.0):
        """Initialize SAR detection limits analysis"""
        self.fc = fc  # Carrier frequency
        self.B = B    # Bandwidth
        self.c = 3e8  # Speed of light
        self.wavelength = self.c / fc
        self.antenna_diameter = antenna_diameter
        
        # Physical constants
        self.k_boltzmann = 1.38e-23  # Boltzmann constant (J/K)
        self.T_system = 300  # System temperature (K)
        
        # System parameters
        self.antenna_area = np.pi * (antenna_diameter/2)**2
        self.antenna_gain = 4 * np.pi * self.antenna_area / (self.wavelength**2)
        self.antenna_gain_db = 10 * np.log10(self.antenna_gain)
        
        # Noise and loss parameters
        self.noise_figure_db = 3.0  # dB
        self.noise_figure = 10**(self.noise_figure_db/10)
        self.system_losses_db = 6.0  # dB (cable, atmospheric, processing losses)
        self.system_losses = 10**(self.system_losses_db/10)
        
        # Detection parameters
        self.min_snr_db = 13.0  # dB (for 90% detection, 10^-6 false alarm)
        self.min_snr_linear = 10**(self.min_snr_db/10)
        
        print(f"SAR Detection Limits Analysis")
        print(f"System Parameters:")
        print(f"  Carrier Frequency: {fc/1e9:.1f} GHz")
        print(f"  Wavelength: {self.wavelength*100:.2f} cm")
        print(f"  Bandwidth: {B/1e6:.1f} MHz")
        print(f"  Antenna Diameter: {antenna_diameter:.1f} m")
        print(f"  Antenna Gain: {self.antenna_gain_db:.1f} dB")
        print(f"  Noise Figure: {self.noise_figure_db:.1f} dB")
        print(f"  System Losses: {self.system_losses_db:.1f} dB")
        print(f"  Min SNR for Detection: {self.min_snr_db:.1f} dB")
    
    def calculate_radar_equation(self, Pt, R, sigma, integration_time=1.0):
        """Calculate received power and SNR using radar equation"""
        
        # Radar equation: Pr = (Pt * G^2 * λ^2 * σ) / ((4π)^3 * R^4 * L)
        numerator = Pt * self.antenna_gain**2 * self.wavelength**2 * sigma
        denominator = (4 * np.pi)**3 * R**4 * self.system_losses
        Pr = numerator / denominator
        
        # Noise power: Pn = k * T * B * F
        noise_power = self.k_boltzmann * self.T_system * self.B * self.noise_figure
        
        # Processing gain from integration
        # Coherent integration: Gain = N (for N pulses)
        # Non-coherent integration: Gain = sqrt(N)
        # Use coherent integration for SAR
        num_pulses = integration_time * 1000  # Assume 1kHz PRF
        processing_gain = num_pulses
        
        # Effective noise power after integration
        effective_noise_power = noise_power / processing_gain
        
        # Signal-to-noise ratio
        SNR_linear = Pr / effective_noise_power
        SNR_db = 10 * np.log10(SNR_linear)
        
        # Detection capability
        detectable = SNR_db >= self.min_snr_db
        
        return {
            'received_power': Pr,
            'noise_power': effective_noise_power,
            'SNR_linear': SNR_linear,
            'SNR_db': SNR_db,
            'detectable': detectable,
            'processing_gain': processing_gain,
            'num_pulses': num_pulses
        }
    
    def calculate_maximum_range(self, Pt, sigma, integration_time=1.0):
        """Calculate maximum detection range for given power and RCS"""
        
        # Rearrange radar equation to solve for maximum range
        # R_max = [(Pt * G^2 * λ^2 * σ * N) / ((4π)^3 * SNR_min * k * T * B * F * L)]^(1/4)
        
        num_pulses = integration_time * 1000
        processing_gain = num_pulses
        
        numerator = Pt * self.antenna_gain**2 * self.wavelength**2 * sigma * processing_gain
        denominator = (4 * np.pi)**3 * self.min_snr_linear * self.k_boltzmann * self.T_system * self.B * self.noise_figure * self.system_losses
        
        R_max = (numerator / denominator)**(1/4)
        
        return R_max
    
    def calculate_minimum_rcs(self, Pt, R, integration_time=1.0):
        """Calculate minimum detectable RCS for given power and range"""
        
        # Rearrange radar equation to solve for minimum RCS
        # σ_min = (SNR_min * k * T * B * F * L * (4π)^3 * R^4) / (Pt * G^2 * λ^2 * N)
        
        num_pulses = integration_time * 1000
        processing_gain = num_pulses
        
        numerator = self.min_snr_linear * self.k_boltzmann * self.T_system * self.B * self.noise_figure * self.system_losses * (4 * np.pi)**3 * R**4
        denominator = Pt * self.antenna_gain**2 * self.wavelength**2 * processing_gain
        
        sigma_min = numerator / denominator
        
        return sigma_min
    
    def analyze_power_vs_range_detection(self):
        """Analyze detection capability vs power and range"""
        print(f"\nAnalyzing Power vs Range Detection Limits...")
        
        # Define parameter ranges
        powers_watts = np.logspace(2, 6, 50)  # 100W to 1MW
        ranges_km = np.linspace(100, 2000, 100)  # 100km to 2000km
        
        # Test different RCS values
        rcs_values = [0.01, 0.1, 1.0, 10.0, 100.0]  # m²
        rcs_labels = ['0.01 m² (Small)', '0.1 m² (Medium)', '1 m² (Large)', 
                     '10 m² (Very Large)', '100 m² (Huge)']
        
        detection_results = {}
        
        for i, (rcs, label) in enumerate(zip(rcs_values, rcs_labels)):
            print(f"  Processing RCS = {rcs} m² ({label})")
            
            # Create detection matrix
            detection_matrix = np.zeros((len(powers_watts), len(ranges_km)))
            snr_matrix = np.zeros((len(powers_watts), len(ranges_km)))
            
            for p_idx, power in enumerate(powers_watts):
                for r_idx, range_m in enumerate(ranges_km * 1e3):  # Convert to meters
                    
                    result = self.calculate_radar_equation(power, range_m, rcs)
                    detection_matrix[p_idx, r_idx] = 1 if result['detectable'] else 0
                    snr_matrix[p_idx, r_idx] = result['SNR_db']
            
            detection_results[label] = {
                'rcs': rcs,
                'detection_matrix': detection_matrix,
                'snr_matrix': snr_matrix
            }
        
        return detection_results, powers_watts, ranges_km
    
    def analyze_integration_time_effects(self):
        """Analyze effect of integration time on detection"""
        print(f"\nAnalyzing Integration Time Effects...")
        
        integration_times = np.logspace(-1, 2, 50)  # 0.1s to 100s
        ranges_km = np.linspace(200, 1500, 100)
        
        # Fixed parameters
        power_watts = 10e3  # 10 kW
        rcs_values = [0.1, 1.0, 10.0]  # m²
        
        integration_results = {}
        
        for rcs in rcs_values:
            detection_matrix = np.zeros((len(integration_times), len(ranges_km)))
            snr_matrix = np.zeros((len(integration_times), len(ranges_km)))
            
            for t_idx, int_time in enumerate(integration_times):
                for r_idx, range_m in enumerate(ranges_km * 1e3):
                    
                    result = self.calculate_radar_equation(power_watts, range_m, rcs, int_time)
                    detection_matrix[t_idx, r_idx] = 1 if result['detectable'] else 0
                    snr_matrix[t_idx, r_idx] = result['SNR_db']
            
            integration_results[f'{rcs} m²'] = {
                'detection_matrix': detection_matrix,
                'snr_matrix': snr_matrix
            }
        
        return integration_results, integration_times, ranges_km
    
    def calculate_detection_boundaries(self):
        """Calculate detection boundaries for different scenarios"""
        print(f"\nCalculating Detection Boundaries...")
        
        # Scenario 1: Maximum range vs power for different RCS
        powers_watts_range = np.logspace(2, 6, 100)
        rcs_values = [0.01, 0.1, 1.0, 10.0, 100.0]
        
        max_ranges = {}
        for rcs in rcs_values:
            ranges = []
            for power in powers_watts_range:
                max_range = self.calculate_maximum_range(power, rcs)
                ranges.append(max_range / 1e3)  # Convert to km
            max_ranges[f'{rcs} m²'] = ranges
        
        # Scenario 2: Minimum RCS vs range for different powers
        ranges_km_rcs = np.linspace(100, 2000, 100)
        power_levels = [1e3, 5e3, 10e3, 50e3, 100e3]  # W
        
        min_rcs_curves = {}
        for power in power_levels:
            rcs_values = []
            for range_km in ranges_km_rcs:
                min_rcs = self.calculate_minimum_rcs(power, range_km * 1e3)
                rcs_values.append(min_rcs)
            min_rcs_curves[f'{power/1e3:.0f} kW'] = rcs_values
        
        return max_ranges, powers_watts_range, min_rcs_curves, ranges_km_rcs
    
    def create_detection_analysis_plots(self, detection_results, powers_watts, ranges_km,
                                      integration_results, integration_times, int_ranges_km,
                                      max_ranges, powers_watts_range, min_rcs_curves, ranges_km_rcs):
        """Create comprehensive detection analysis plots"""
        fig = plt.figure(figsize=(20, 16))
        
        # Plot 1: Detection Matrix for Medium RCS (1 m²)
        ax1 = plt.subplot(2, 4, 1)
        
        medium_rcs_data = detection_results['1 m² (Large)']
        detection_matrix = medium_rcs_data['detection_matrix']
        
        extent = [ranges_km[0], ranges_km[-1], powers_watts[0]/1e3, powers_watts[-1]/1e3]
        im1 = ax1.imshow(detection_matrix, aspect='auto', origin='lower', extent=extent,
                        cmap='RdYlGn', alpha=0.8)
        
        ax1.set_xlabel('Range (km)', fontsize=12)
        ax1.set_ylabel('Transmit Power (kW)', fontsize=12)
        ax1.set_title('Detection Capability\nRCS = 1 m²', fontweight='bold')
        ax1.set_yscale('log')
        
        # Add detection boundary line
        detection_boundary_ranges = []
        detection_boundary_powers = []
        for p_idx, power in enumerate(powers_watts):
            for r_idx, range_val in enumerate(ranges_km):
                if detection_matrix[p_idx, r_idx] == 1:
                    if r_idx == 0 or detection_matrix[p_idx, r_idx-1] == 0:
                        detection_boundary_ranges.append(range_val)
                        detection_boundary_powers.append(power/1e3)
                        break
        
        if detection_boundary_ranges:
            ax1.plot(detection_boundary_ranges, detection_boundary_powers, 'r-', 
                    linewidth=3, label='Detection Boundary')
            ax1.legend()
        
        plt.colorbar(im1, ax=ax1, label='Detectable (1=Yes, 0=No)')
        
        # Plot 2: SNR Contours for Medium RCS
        ax2 = plt.subplot(2, 4, 2)
        
        snr_matrix = medium_rcs_data['snr_matrix']
        
        # Create contour plot
        R_grid, P_grid = np.meshgrid(ranges_km, powers_watts/1e3)
        
        contour_levels = [0, 5, 10, 13, 15, 20, 25, 30]
        cs = ax2.contour(R_grid, P_grid, snr_matrix, levels=contour_levels, colors='black', alpha=0.6)
        ax2.clabel(cs, inline=True, fontsize=8, fmt='%d dB')
        
        # Fill contours
        cs_filled = ax2.contourf(R_grid, P_grid, snr_matrix, levels=contour_levels, 
                                cmap='viridis', alpha=0.7)
        
        # Highlight detection threshold
        ax2.contour(R_grid, P_grid, snr_matrix, levels=[13], colors='red', linewidths=3)
        
        ax2.set_xlabel('Range (km)', fontsize=12)
        ax2.set_ylabel('Transmit Power (kW)', fontsize=12)
        ax2.set_title('SNR Contours\nRCS = 1 m² (Red line = 13 dB threshold)', fontweight='bold')
        ax2.set_yscale('log')
        
        plt.colorbar(cs_filled, ax=ax2, label='SNR (dB)')
        
        # Plot 3: Maximum Range vs Power for Different RCS
        ax3 = plt.subplot(2, 4, 3)
        
        colors = ['red', 'orange', 'green', 'blue', 'purple']
        for i, (rcs_label, ranges) in enumerate(max_ranges.items()):
            ax3.plot(powers_watts_range/1e3, ranges, color=colors[i], linewidth=2, 
                    label=f'RCS = {rcs_label}')
        
        ax3.set_xlabel('Transmit Power (kW)', fontsize=12)
        ax3.set_ylabel('Maximum Range (km)', fontsize=12)
        ax3.set_title('Maximum Detection Range\nvs Transmit Power', fontweight='bold')
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Plot 4: Minimum RCS vs Range for Different Powers
        ax4 = plt.subplot(2, 4, 4)
        
        for i, (power_label, rcs_values) in enumerate(min_rcs_curves.items()):
            ax4.plot(ranges_km_rcs, rcs_values, color=colors[i], linewidth=2, 
                    label=f'Power = {power_label}')
        
        ax4.set_xlabel('Range (km)', fontsize=12)
        ax4.set_ylabel('Minimum Detectable RCS (m²)', fontsize=12)
        ax4.set_title('Minimum Detectable RCS\nvs Range', fontweight='bold')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # Add reference lines for typical targets
        ax4.axhline(0.01, color='red', linestyle='--', alpha=0.7, label='Small aircraft')
        ax4.axhline(1.0, color='orange', linestyle='--', alpha=0.7, label='Large aircraft')
        ax4.axhline(100.0, color='blue', linestyle='--', alpha=0.7, label='Ship')
        
        # Plot 5: Integration Time Effects
        ax5 = plt.subplot(2, 4, 5)
        
        # Show detection boundary vs integration time for medium RCS
        int_data = integration_results['1.0 m²']
        int_detection_matrix = int_data['detection_matrix']
        
        extent_int = [int_ranges_km[0], int_ranges_km[-1], 
                     integration_times[0], integration_times[-1]]
        im5 = ax5.imshow(int_detection_matrix, aspect='auto', origin='lower', 
                        extent=extent_int, cmap='RdYlGn', alpha=0.8)
        
        ax5.set_xlabel('Range (km)', fontsize=12)
        ax5.set_ylabel('Integration Time (s)', fontsize=12)
        ax5.set_title('Detection vs Integration Time\nRCS = 1 m², Power = 10 kW', fontweight='bold')
        ax5.set_yscale('log')
        
        plt.colorbar(im5, ax=ax5, label='Detectable')
        
        # Plot 6: Processing Gain Effects
        ax6 = plt.subplot(2, 4, 6)
        
        # Calculate processing gain vs integration time
        processing_gains_db = 10 * np.log10(integration_times * 1000)  # Assume 1kHz PRF
        
        ax6.plot(integration_times, processing_gains_db, 'b-', linewidth=3, 
                label='Coherent Integration')
        
        # Non-coherent integration for comparison
        processing_gains_noncoherent_db = 10 * np.log10(np.sqrt(integration_times * 1000))
        ax6.plot(integration_times, processing_gains_noncoherent_db, 'r--', linewidth=2, 
                label='Non-coherent Integration')
        
        ax6.set_xlabel('Integration Time (s)', fontsize=12)
        ax6.set_ylabel('Processing Gain (dB)', fontsize=12)
        ax6.set_title('Processing Gain vs\nIntegration Time', fontweight='bold')
        ax6.set_xscale('log')
        ax6.grid(True, alpha=0.3)
        ax6.legend()
        
        # Plot 7: Target Classification by RCS
        ax7 = plt.subplot(2, 4, 7)
        
        # Typical RCS values for different targets
        targets = ['Stealth Aircraft', 'Small Aircraft', 'Fighter Aircraft', 
                  'Large Aircraft', 'Small Ship', 'Large Ship', 'Building']
        rcs_typical = [0.001, 0.01, 0.1, 1.0, 10.0, 1000.0, 10000.0]
        colors_targets = ['black', 'red', 'orange', 'green', 'blue', 'purple', 'brown']
        
        # Calculate maximum range for each target type (10 kW power)
        power_ref = 10e3  # 10 kW
        max_ranges_targets = []
        for rcs in rcs_typical:
            max_range = self.calculate_maximum_range(power_ref, rcs) / 1e3  # km
            max_ranges_targets.append(max_range)
        
        bars = ax7.barh(range(len(targets)), max_ranges_targets, color=colors_targets, alpha=0.7)
        ax7.set_yticks(range(len(targets)))
        ax7.set_yticklabels(targets)
        ax7.set_xlabel('Maximum Detection Range (km)', fontsize=12)
        ax7.set_title('Detection Range by Target Type\n(10 kW transmit power)', fontweight='bold')
        ax7.grid(True, alpha=0.3, axis='x')
        
        # Add RCS labels on bars
        for i, (bar, rcs) in enumerate(zip(bars, rcs_typical)):
            ax7.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
                    f'{rcs} m²', va='center', fontsize=9)
        
        # Plot 8: Radar Equation Components
        ax8 = plt.subplot(2, 4, 8)
        
        # Show how different components affect received power
        range_ref = 500e3  # 500 km
        rcs_ref = 1.0  # 1 m²
        power_ref = 10e3  # 10 kW
        
        # Vary each parameter individually
        param_multipliers = np.logspace(-1, 1, 50)  # 0.1x to 10x
        
        # Power variation
        power_effect = []
        for mult in param_multipliers:
            result = self.calculate_radar_equation(power_ref * mult, range_ref, rcs_ref)
            power_effect.append(result['SNR_db'])
        
        # Range variation (R^4 dependence)
        range_effect = []
        for mult in param_multipliers:
            result = self.calculate_radar_equation(power_ref, range_ref * mult, rcs_ref)
            range_effect.append(result['SNR_db'])
        
        # RCS variation
        rcs_effect = []
        for mult in param_multipliers:
            result = self.calculate_radar_equation(power_ref, range_ref, rcs_ref * mult)
            rcs_effect.append(result['SNR_db'])
        
        ax8.plot(param_multipliers, power_effect, 'b-', linewidth=2, label='Power (Pt)')
        ax8.plot(param_multipliers, range_effect, 'r-', linewidth=2, label='Range (R⁴)')
        ax8.plot(param_multipliers, rcs_effect, 'g-', linewidth=2, label='RCS (σ)')
        
        ax8.axhline(self.min_snr_db, color='black', linestyle='--', linewidth=2, 
                   label='Detection Threshold')
        
        ax8.set_xlabel('Parameter Multiplier', fontsize=12)
        ax8.set_ylabel('SNR (dB)', fontsize=12)
        ax8.set_title('Radar Equation\nParameter Sensitivity', fontweight='bold')
        ax8.set_xscale('log')
        ax8.grid(True, alpha=0.3)
        ax8.legend()
        
        plt.suptitle('SAR Detection Limits Analysis\nPower, Range, and RCS Effects on Target Detectability', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('../output/sar_detection_limits_analysis.png', dpi=300, bbox_inches='tight')
        print("\nSAR detection limits analysis plots saved to: ../output/sar_detection_limits_analysis.png")
        plt.close()
    
    def run_detection_analysis(self):
        """Run comprehensive detection limits analysis"""
        print("SAR DETECTION LIMITS ANALYSIS")
        print("="*80)
        print("Analyzing target detectability based on power, range, and RCS")
        print("="*80)
        
        # Create output directory
        os.makedirs('../output', exist_ok=True)
        
        # Analyze power vs range detection
        detection_results, powers_watts, ranges_km = self.analyze_power_vs_range_detection()
        
        # Analyze integration time effects
        integration_results, integration_times, int_ranges_km = self.analyze_integration_time_effects()
        
        # Calculate detection boundaries
        max_ranges, powers_for_range, min_rcs_curves, ranges_for_rcs = self.calculate_detection_boundaries()
        
        # Create comprehensive plots
        self.create_detection_analysis_plots(detection_results, powers_watts, ranges_km,
                                           integration_results, integration_times, int_ranges_km,
                                           max_ranges, powers_for_range, min_rcs_curves, ranges_for_rcs)
        
        # Analysis summary
        print(f"\n" + "="*80)
        print("SAR DETECTION LIMITS ANALYSIS SUMMARY")
        print("="*80)
        
        print(f"\n1. RADAR EQUATION FUNDAMENTALS:")
        print(f"   Received Power: Pr = (Pt × G² × λ² × σ) / ((4π)³ × R⁴ × L)")
        print(f"   SNR: SNR = Pr / (k × T × B × F)")
        print(f"   Processing Gain: G_proc = N_pulses (coherent integration)")
        
        print(f"\n2. SYSTEM PARAMETERS:")
        print(f"   Antenna Gain: {self.antenna_gain_db:.1f} dB")
        print(f"   System Temperature: {self.T_system:.0f} K")
        print(f"   Noise Figure: {self.noise_figure_db:.1f} dB")
        print(f"   System Losses: {self.system_losses_db:.1f} dB")
        print(f"   Detection Threshold: {self.min_snr_db:.1f} dB SNR")
        
        print(f"\n3. DETECTION RANGE EXAMPLES:")
        # Calculate some example cases
        example_powers = [1e3, 10e3, 100e3]  # 1, 10, 100 kW
        example_rcs = [0.01, 1.0, 100.0]  # Small, medium, large targets
        
        print(f"   Power (kW) | RCS (m²) | Max Range (km)")
        print(f"   " + "-"*40)
        for power in example_powers:
            for rcs in example_rcs:
                max_range = self.calculate_maximum_range(power, rcs) / 1e3
                print(f"   {power/1e3:8.0f}   |  {rcs:6.2f}  |   {max_range:8.0f}")
        
        print(f"\n4. MINIMUM DETECTABLE RCS:")
        print(f"   Range (km) | Power (kW) | Min RCS (m²)")
        print(f"   " + "-"*40)
        example_ranges = [200e3, 500e3, 1000e3]  # 200, 500, 1000 km
        for range_m in example_ranges:
            for power in [10e3, 50e3]:  # 10, 50 kW
                min_rcs = self.calculate_minimum_rcs(power, range_m)
                print(f"   {range_m/1e3:7.0f}    |   {power/1e3:6.0f}   |   {min_rcs:8.4f}")
        
        print(f"\n5. KEY INSIGHTS:")
        print(f"   R⁴ Dependence:")
        print(f"   - Doubling range reduces received power by 16x (-12 dB)")
        print(f"   - Detection range scales as P^(1/4) with power")
        
        print(f"   Integration Benefits:")
        print(f"   - 10x integration time → 10 dB processing gain")
        print(f"   - Coherent integration: Gain = N_pulses")
        print(f"   - Non-coherent integration: Gain = √N_pulses")
        
        print(f"   Practical Limitations:")
        print(f"   - Stealth targets (0.001 m²): Very short range")
        print(f"   - Small aircraft (0.01 m²): Limited range")
        print(f"   - Large targets (100+ m²): Long range detection")
        
        print(f"\n6. DESIGN TRADE-OFFS:")
        print(f"   High Power Systems:")
        print(f"   - Pros: Long detection range, small target capability")
        print(f"   - Cons: Size, weight, power consumption, cost")
        
        print(f"   Long Integration Systems:")
        print(f"   - Pros: Improved SNR, better detection")
        print(f"   - Cons: Slower coverage, motion sensitivity")
        
        print(f"   Large Antenna Systems:")
        print(f"   - Pros: High gain, better resolution")
        print(f"   - Cons: Mechanical complexity, pointing accuracy")
        
        # Save comprehensive results
        analysis_results = {
            'timestamp': datetime.now().isoformat(),
            'analysis_type': 'sar_detection_limits',
            'system_parameters': {
                'carrier_frequency': self.fc,
                'wavelength': self.wavelength,
                'bandwidth': self.B,
                'antenna_diameter': self.antenna_diameter,
                'antenna_gain_db': self.antenna_gain_db,
                'noise_figure_db': self.noise_figure_db,
                'system_losses_db': self.system_losses_db,
                'min_snr_db': self.min_snr_db,
                'system_temperature': self.T_system
            },
            'radar_equation': {
                'received_power': 'Pr = (Pt × G² × λ² × σ) / ((4π)³ × R⁴ × L)',
                'snr': 'SNR = Pr / (k × T × B × F × N_pulses)',
                'max_range': 'R_max = [(Pt × G² × λ² × σ × N) / ((4π)³ × SNR_min × k × T × B × F × L)]^(1/4)',
                'min_rcs': 'σ_min = (SNR_min × k × T × B × F × L × (4π)³ × R⁴) / (Pt × G² × λ² × N)'
            },
            'key_insights': {
                'range_dependence': 'R⁴ - very strong',
                'power_scaling': 'Detection range ∝ Power^(1/4)',
                'integration_gain': 'Coherent: N_pulses, Non-coherent: √N_pulses',
                'rcs_impact': 'Linear with received power'
            }
        }
        
        with open('../output/sar_detection_limits_results.json', 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        print(f"\nDetailed detection analysis results saved to: ../output/sar_detection_limits_results.json")
        
        return detection_results, integration_results, max_ranges, min_rcs_curves, analysis_results

def main():
    """Main execution"""
    analysis = SARDetectionLimitsAnalysis()
    detection_results, integration_results, max_ranges, min_rcs_curves, results = analysis.run_detection_analysis()
    return analysis, detection_results, integration_results, max_ranges, min_rcs_curves, results

if __name__ == "__main__":
    analysis, detection_results, integration_results, max_ranges, min_rcs_curves, results = main()
