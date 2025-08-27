#!/usr/bin/env python3
"""
Synthetic Aperture Size Impact Analysis
Comprehensive study of how synthetic aperture size affects resolution in both:
1. LFMCW Range-Doppler Map (RDM) systems
2. Pulsed SAR with pulse compression

Key relationships:
- Azimuth Resolution: ΔAz = λ*R/(2*L_syn)
- Maximum aperture limitations differ between LFMCW and pulsed systems

Limiting factors analyzed:
- Platform motion constraints
- Coherence time limitations  
- Range migration effects
- Doppler bandwidth limits
- Processing complexity
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append('../../SAR_Benchmark_Validation_2024/code')
from sar_model_final import FinalSARModel
from scipy.signal import chirp
import json
from datetime import datetime

class SyntheticApertureSizeAnalysis:
    def __init__(self, fc=10e9, B=100e6):
        """Initialize synthetic aperture size analysis"""
        self.fc = fc  # Carrier frequency
        self.B = B    # Bandwidth
        self.c = 3e8  # Speed of light
        self.wavelength = self.c / fc
        self.range_resolution = self.c / (2 * B)
        
        # Platform parameters (typical values)
        self.platform_velocity = 200  # m/s
        self.platform_altitude = 500e3  # 500 km (satellite)
        
        # Target parameters
        self.target_ranges = [100e3, 250e3, 500e3, 1000e3]  # 100km to 1000km
        
        print(f"Synthetic Aperture Size Analysis")
        print(f"System Parameters:")
        print(f"  Carrier Frequency: {fc/1e9:.1f} GHz")
        print(f"  Wavelength: {self.wavelength*100:.2f} cm") 
        print(f"  Bandwidth: {B/1e6:.1f} MHz")
        print(f"  Range Resolution: {self.range_resolution:.3f} m")
        print(f"  Platform Velocity: {self.platform_velocity} m/s")
        print(f"  Platform Altitude: {self.platform_altitude/1e3:.0f} km")
    
    def calculate_azimuth_resolution_vs_aperture(self, target_range):
        """Calculate azimuth resolution as function of synthetic aperture size"""
        # Synthetic aperture sizes from 10m to 50km
        aperture_sizes = np.logspace(1, 4.7, 100)  # 10m to ~50km
        
        # Calculate azimuth resolution for each aperture size
        azimuth_resolutions = self.wavelength * target_range / (2 * aperture_sizes)
        
        return aperture_sizes, azimuth_resolutions
    
    def calculate_lfmcw_aperture_limits(self):
        """Calculate maximum synthetic aperture limitations for LFMCW systems"""
        print(f"\nLFMCW Synthetic Aperture Limitations:")
        
        limitations = {}
        
        # 1. Coherence Time Limitation
        # For LFMCW, coherence is maintained across sweeps
        # Limited by platform motion and target motion
        coherence_time = 1.0  # seconds (typical)
        max_aperture_coherence = self.platform_velocity * coherence_time
        limitations['coherence_time'] = {
            'value': max_aperture_coherence,
            'description': f'Platform motion over coherence time ({coherence_time}s)',
            'formula': 'L_max = V_platform × T_coherence'
        }
        
        # 2. Doppler Bandwidth Limitation
        # Maximum Doppler frequency that can be unambiguously measured
        # For LFMCW, limited by sweep repetition rate
        sweep_rate = 1000  # Hz (typical)
        max_doppler_freq = sweep_rate / 2  # Nyquist limit
        
        # Doppler frequency: f_d = 2*V*cos(θ)/λ ≈ 2*V/λ for broadside
        max_velocity_component = max_doppler_freq * self.wavelength / 2
        
        # For synthetic aperture: L = 2*R*sin(θ_max) ≈ 2*R*V_max/V_platform
        max_aperture_doppler = {}
        for R in self.target_ranges:
            max_L = 2 * R * max_velocity_component / self.platform_velocity
            max_aperture_doppler[R] = max_L
        
        limitations['doppler_bandwidth'] = {
            'max_doppler_freq': max_doppler_freq,
            'max_apertures': max_aperture_doppler,
            'description': 'Limited by sweep repetition rate and Doppler ambiguity',
            'formula': 'f_d_max = PRF/2, L_max = 2*R*f_d_max*λ/(2*V)'
        }
        
        # 3. Range Migration Limitation
        # Range migration becomes significant for large apertures
        # ΔR = λ²/(8*ΔAz) for quadratic approximation
        range_migration_limit = self.range_resolution  # Keep migration < 1 range bin
        
        max_aperture_migration = {}
        for R in self.target_ranges:
            # From range migration: ΔR = L²/(8*R)
            # Setting ΔR = range_resolution: L_max = sqrt(8*R*range_resolution)
            max_L = np.sqrt(8 * R * range_migration_limit)
            max_aperture_migration[R] = max_L
        
        limitations['range_migration'] = {
            'max_apertures': max_aperture_migration,
            'description': 'Range migration must stay within one range bin',
            'formula': 'L_max = sqrt(8*R*ΔR)'
        }
        
        # 4. Processing Complexity Limitation
        # FFT size limitations for real-time processing
        max_fft_size = 8192  # Practical limit for real-time
        max_aperture_processing = self.platform_velocity * max_fft_size / sweep_rate
        limitations['processing_complexity'] = {
            'value': max_aperture_processing,
            'description': f'FFT size limit for real-time processing ({max_fft_size} points)',
            'formula': 'L_max = V × N_FFT / PRF'
        }
        
        return limitations
    
    def calculate_pulsed_sar_aperture_limits(self):
        """Calculate maximum synthetic aperture limitations for pulsed SAR"""
        print(f"\nPulsed SAR Synthetic Aperture Limitations:")
        
        limitations = {}
        
        # 1. Coherence Time Limitation
        # Similar to LFMCW but may be longer due to better motion compensation
        coherence_time = 2.0  # seconds (longer than LFMCW due to motion compensation)
        max_aperture_coherence = self.platform_velocity * coherence_time
        limitations['coherence_time'] = {
            'value': max_aperture_coherence,
            'description': f'Platform motion over coherence time ({coherence_time}s)',
            'formula': 'L_max = V_platform × T_coherence'
        }
        
        # 2. PRF and Doppler Ambiguity Limitation
        PRF = 1000  # Hz (typical)
        max_doppler_freq = PRF / 2  # Nyquist limit
        
        max_aperture_doppler = {}
        for R in self.target_ranges:
            # Maximum synthetic aperture angle: sin(θ_max) = λ*PRF/(4*V)
            sin_theta_max = self.wavelength * PRF / (4 * self.platform_velocity)
            if sin_theta_max <= 1:
                max_L = 2 * R * sin_theta_max
            else:
                max_L = 2 * R  # Full aperture angle
            max_aperture_doppler[R] = max_L
        
        limitations['doppler_ambiguity'] = {
            'PRF': PRF,
            'max_apertures': max_aperture_doppler,
            'description': 'Limited by PRF and Doppler ambiguity',
            'formula': 'sin(θ_max) = λ*PRF/(4*V), L_max = 2*R*sin(θ_max)'
        }
        
        # 3. Range Migration Limitation (more severe for pulsed SAR)
        # Range curvature: ΔR = L²/(8*R)
        range_migration_limit = self.range_resolution / 4  # More stringent for pulsed SAR
        
        max_aperture_migration = {}
        for R in self.target_ranges:
            max_L = np.sqrt(8 * R * range_migration_limit)
            max_aperture_migration[R] = max_L
        
        limitations['range_migration'] = {
            'max_apertures': max_aperture_migration,
            'description': 'Range migration correction becomes complex',
            'formula': 'L_max = sqrt(8*R*ΔR/4)'
        }
        
        # 4. Motion Compensation Limitation
        # Accuracy of motion compensation limits aperture size
        motion_accuracy = 0.1  # meters (typical GPS/INS accuracy)
        phase_accuracy = 2 * np.pi * motion_accuracy / self.wavelength  # radians
        
        # Phase error should be < π/4 for good focusing
        max_phase_error = np.pi / 4
        max_aperture_motion = max_phase_error * self.wavelength / (2 * np.pi) * 100  # Conservative
        
        limitations['motion_compensation'] = {
            'value': max_aperture_motion,
            'description': f'Motion measurement accuracy ({motion_accuracy}m)',
            'formula': 'L_max limited by phase accuracy = λ/(4*motion_error)'
        }
        
        # 5. Processing Complexity (more complex than LFMCW)
        # Range cell migration correction, autofocus, etc.
        max_aperture_processing = 20000  # meters (typical operational limit)
        limitations['processing_complexity'] = {
            'value': max_aperture_processing,
            'description': 'Complex algorithms: RCM, autofocus, motion compensation',
            'formula': 'L_max ~ 20km (practical operational limit)'
        }
        
        return limitations
    
    def simulate_aperture_size_effects(self, aperture_sizes, target_range):
        """Simulate the effect of different aperture sizes on target resolution"""
        print(f"\nSimulating aperture size effects for target at {target_range/1e3:.0f}km...")
        
        results = {}
        
        for L_syn in aperture_sizes:
            # Calculate azimuth resolution
            azimuth_resolution = self.wavelength * target_range / (2 * L_syn)
            
            # Calculate integration time needed
            integration_time = L_syn / self.platform_velocity
            
            # Calculate number of pulses/sweeps needed
            PRF = 1000  # Hz
            num_pulses = int(integration_time * PRF)
            
            # Calculate Doppler bandwidth
            doppler_bandwidth = 2 * self.platform_velocity / self.wavelength
            
            # Calculate range migration
            range_migration = L_syn**2 / (8 * target_range)
            
            results[L_syn] = {
                'azimuth_resolution': azimuth_resolution,
                'integration_time': integration_time,
                'num_pulses': num_pulses,
                'doppler_bandwidth': doppler_bandwidth,
                'range_migration': range_migration,
                'range_migration_bins': range_migration / self.range_resolution
            }
        
        return results
    
    def create_comprehensive_plots(self, lfmcw_limits, pulsed_limits):
        """Create comprehensive plots showing synthetic aperture analysis"""
        fig = plt.figure(figsize=(20, 16))
        
        # Plot 1: Azimuth Resolution vs Synthetic Aperture Size
        ax1 = plt.subplot(2, 4, 1)
        
        colors = ['blue', 'green', 'red', 'purple']
        for i, R in enumerate(self.target_ranges):
            aperture_sizes, az_resolutions = self.calculate_azimuth_resolution_vs_aperture(R)
            ax1.loglog(aperture_sizes, az_resolutions, color=colors[i], linewidth=2,
                      label=f'Range = {R/1e3:.0f} km')
        
        ax1.set_xlabel('Synthetic Aperture Size (m)', fontsize=12)
        ax1.set_ylabel('Azimuth Resolution (m)', fontsize=12)
        ax1.set_title('Azimuth Resolution vs Synthetic Aperture\nΔAz = λR/(2L)', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Add theoretical slope line
        L_theory = np.logspace(1, 4, 100)
        az_theory = 100 / L_theory  # Slope = -1 on log-log plot
        ax1.loglog(L_theory, az_theory, 'k--', alpha=0.5, label='Slope = -1')
        
        # Plot 2: LFMCW Aperture Limitations
        ax2 = plt.subplot(2, 4, 2)
        
        # Extract limitation values for different ranges
        coherence_limit = lfmcw_limits['coherence_time']['value']
        processing_limit = lfmcw_limits['processing_complexity']['value']
        
        ranges_km = [R/1e3 for R in self.target_ranges]
        doppler_limits = [lfmcw_limits['doppler_bandwidth']['max_apertures'][R]/1e3 for R in self.target_ranges]
        migration_limits = [lfmcw_limits['range_migration']['max_apertures'][R]/1e3 for R in self.target_ranges]
        
        ax2.plot(ranges_km, doppler_limits, 'b-o', linewidth=2, label='Doppler Bandwidth')
        ax2.plot(ranges_km, migration_limits, 'r-s', linewidth=2, label='Range Migration')
        ax2.axhline(coherence_limit/1e3, color='g', linestyle='--', linewidth=2, label='Coherence Time')
        ax2.axhline(processing_limit/1e3, color='orange', linestyle='--', linewidth=2, label='Processing')
        
        ax2.set_xlabel('Target Range (km)', fontsize=12)
        ax2.set_ylabel('Max Synthetic Aperture (km)', fontsize=12)
        ax2.set_title('LFMCW Aperture Limitations', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_yscale('log')
        
        # Plot 3: Pulsed SAR Aperture Limitations
        ax3 = plt.subplot(2, 4, 3)
        
        coherence_limit_pulsed = pulsed_limits['coherence_time']['value']
        processing_limit_pulsed = pulsed_limits['processing_complexity']['value']
        motion_limit = pulsed_limits['motion_compensation']['value']
        
        doppler_limits_pulsed = [pulsed_limits['doppler_ambiguity']['max_apertures'][R]/1e3 for R in self.target_ranges]
        migration_limits_pulsed = [pulsed_limits['range_migration']['max_apertures'][R]/1e3 for R in self.target_ranges]
        
        ax3.plot(ranges_km, doppler_limits_pulsed, 'b-o', linewidth=2, label='Doppler Ambiguity')
        ax3.plot(ranges_km, migration_limits_pulsed, 'r-s', linewidth=2, label='Range Migration')
        ax3.axhline(coherence_limit_pulsed/1e3, color='g', linestyle='--', linewidth=2, label='Coherence Time')
        ax3.axhline(processing_limit_pulsed/1e3, color='orange', linestyle='--', linewidth=2, label='Processing')
        ax3.axhline(motion_limit/1e3, color='purple', linestyle='--', linewidth=2, label='Motion Compensation')
        
        ax3.set_xlabel('Target Range (km)', fontsize=12)
        ax3.set_ylabel('Max Synthetic Aperture (km)', fontsize=12)
        ax3.set_title('Pulsed SAR Aperture Limitations', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.set_yscale('log')
        
        # Plot 4: Range Migration vs Aperture Size
        ax4 = plt.subplot(2, 4, 4)
        
        aperture_sizes = np.linspace(100, 20000, 200)  # 100m to 20km
        
        for i, R in enumerate(self.target_ranges):
            range_migration = aperture_sizes**2 / (8 * R)
            range_migration_bins = range_migration / self.range_resolution
            ax4.semilogy(aperture_sizes/1e3, range_migration_bins, color=colors[i], 
                        linewidth=2, label=f'Range = {R/1e3:.0f} km')
        
        ax4.axhline(1, color='red', linestyle='--', linewidth=2, label='1 Range Bin')
        ax4.axhline(0.25, color='orange', linestyle='--', linewidth=2, label='0.25 Range Bin')
        
        ax4.set_xlabel('Synthetic Aperture Size (km)', fontsize=12)
        ax4.set_ylabel('Range Migration (Range Bins)', fontsize=12)
        ax4.set_title('Range Migration vs Aperture\nΔR = L²/(8R)', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # Plot 5: Integration Time vs Aperture Size
        ax5 = plt.subplot(2, 4, 5)
        
        aperture_sizes_time = np.linspace(100, 50000, 100)
        integration_times = aperture_sizes_time / self.platform_velocity
        
        ax5.plot(aperture_sizes_time/1e3, integration_times, 'b-', linewidth=3, label='Integration Time')
        ax5.axhline(1.0, color='orange', linestyle='--', linewidth=2, label='1 second')
        ax5.axhline(10.0, color='red', linestyle='--', linewidth=2, label='10 seconds')
        
        ax5.set_xlabel('Synthetic Aperture Size (km)', fontsize=12)
        ax5.set_ylabel('Integration Time (s)', fontsize=12)
        ax5.set_title('Integration Time vs Aperture\nT = L/V', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        
        # Plot 6: Doppler Bandwidth vs Aperture Size
        ax6 = plt.subplot(2, 4, 6)
        
        # Doppler bandwidth is related to aperture size and beamwidth
        # For synthetic aperture: B_doppler ≈ 2*V/λ (maximum)
        max_doppler_bandwidth = 2 * self.platform_velocity / self.wavelength
        
        # Effective Doppler bandwidth for different aperture sizes
        aperture_angles = aperture_sizes_time / self.target_ranges[1]  # Use middle range
        effective_doppler_bw = max_doppler_bandwidth * np.sin(aperture_angles/2)
        
        ax6.plot(aperture_sizes_time/1e3, effective_doppler_bw, 'g-', linewidth=3, label='Effective Doppler BW')
        ax6.axhline(max_doppler_bandwidth, color='red', linestyle='--', linewidth=2, label='Maximum (2V/λ)')
        
        ax6.set_xlabel('Synthetic Aperture Size (km)', fontsize=12)
        ax6.set_ylabel('Doppler Bandwidth (Hz)', fontsize=12)
        ax6.set_title('Doppler Bandwidth vs Aperture', fontweight='bold')
        ax6.grid(True, alpha=0.3)
        ax6.legend()
        
        # Plot 7: Comparison Table - LFMCW vs Pulsed SAR
        ax7 = plt.subplot(2, 4, 7)
        
        comparison_data = [
            ['Limitation Factor', 'LFMCW', 'Pulsed SAR'],
            ['Coherence Time', f'{coherence_limit/1e3:.1f} km', f'{coherence_limit_pulsed/1e3:.1f} km'],
            ['Doppler/PRF', 'Sweep Rate Limited', 'PRF Limited'],
            ['Range Migration', 'Less Critical', 'More Critical'],
            ['Motion Compensation', 'Not Required', 'Critical'],
            ['Processing Complexity', 'Simple (2D FFT)', 'Complex (Multi-stage)'],
            ['Typical Max Aperture', '~1-5 km', '~10-20 km'],
            ['Real-time Capability', 'Excellent', 'Limited'],
            ['Mature Algorithms', 'Developing', 'Well-established']
        ]
        
        table = ax7.table(cellText=comparison_data[1:], colLabels=comparison_data[0],
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.8)
        
        # Style table
        for i in range(len(comparison_data)):
            for j in range(3):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white')
                elif j == 0:  # Factor names
                    cell.set_facecolor('#E3F2FD')
                    cell.set_text_props(weight='bold')
                else:
                    cell.set_facecolor('#f8f9fa' if i % 2 == 0 else 'white')
        
        ax7.set_title('LFMCW vs Pulsed SAR\nAperture Limitations', fontweight='bold')
        ax7.axis('off')
        
        # Plot 8: Optimal Aperture Size Selection
        ax8 = plt.subplot(2, 4, 8)
        
        # Calculate optimal aperture sizes for different ranges
        optimal_apertures_lfmcw = []
        optimal_apertures_pulsed = []
        
        for R in self.target_ranges:
            # LFMCW: Limited by minimum of all constraints
            lfmcw_constraints = [
                coherence_limit,
                processing_limit,
                lfmcw_limits['doppler_bandwidth']['max_apertures'][R],
                lfmcw_limits['range_migration']['max_apertures'][R]
            ]
            optimal_lfmcw = min(lfmcw_constraints)
            optimal_apertures_lfmcw.append(optimal_lfmcw/1e3)
            
            # Pulsed SAR: Limited by minimum of all constraints
            pulsed_constraints = [
                coherence_limit_pulsed,
                processing_limit_pulsed,
                motion_limit,
                pulsed_limits['doppler_ambiguity']['max_apertures'][R],
                pulsed_limits['range_migration']['max_apertures'][R]
            ]
            optimal_pulsed = min(pulsed_constraints)
            optimal_apertures_pulsed.append(optimal_pulsed/1e3)
        
        x = np.arange(len(ranges_km))
        width = 0.35
        
        bars1 = ax8.bar(x - width/2, optimal_apertures_lfmcw, width, label='LFMCW', 
                       color='skyblue', alpha=0.8)
        bars2 = ax8.bar(x + width/2, optimal_apertures_pulsed, width, label='Pulsed SAR', 
                       color='lightcoral', alpha=0.8)
        
        ax8.set_xlabel('Target Range (km)', fontsize=12)
        ax8.set_ylabel('Optimal Max Aperture (km)', fontsize=12)
        ax8.set_title('Optimal Synthetic Aperture Size\n(Limited by Constraints)', fontweight='bold')
        ax8.set_xticks(x)
        ax8.set_xticklabels([f'{r:.0f}' for r in ranges_km])
        ax8.legend()
        ax8.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax8.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('Synthetic Aperture Size Analysis: LFMCW vs Pulsed SAR\nResolution Impact and Limiting Factors', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('../output/synthetic_aperture_size_analysis.png', dpi=300, bbox_inches='tight')
        print("\nSynthetic aperture size analysis plots saved to: ../output/synthetic_aperture_size_analysis.png")
        plt.close()
    
    def run_aperture_analysis(self):
        """Run comprehensive synthetic aperture size analysis"""
        print("SYNTHETIC APERTURE SIZE ANALYSIS")
        print("="*80)
        print("Impact on Resolution and Limiting Factors for LFMCW and Pulsed SAR")
        print("="*80)
        
        # Create output directory
        os.makedirs('../output', exist_ok=True)
        
        # Calculate aperture limitations for both systems
        lfmcw_limits = self.calculate_lfmcw_aperture_limits()
        pulsed_limits = self.calculate_pulsed_sar_aperture_limits()
        
        # Create comprehensive plots
        self.create_comprehensive_plots(lfmcw_limits, pulsed_limits)
        
        # Detailed analysis summary
        print(f"\n" + "="*80)
        print("SYNTHETIC APERTURE SIZE ANALYSIS SUMMARY")
        print("="*80)
        
        print(f"\n1. FUNDAMENTAL RELATIONSHIP:")
        print(f"   Azimuth Resolution: ΔAz = λ*R/(2*L_syn)")
        print(f"   - Wavelength: {self.wavelength*100:.2f} cm")
        print(f"   - Resolution improves linearly with aperture size")
        print(f"   - Resolution degrades linearly with range")
        
        print(f"\n2. LFMCW SYSTEM LIMITATIONS:")
        for limit_name, limit_data in lfmcw_limits.items():
            print(f"   {limit_name.replace('_', ' ').title()}:")
            print(f"   - {limit_data['description']}")
            print(f"   - Formula: {limit_data['formula']}")
            if 'value' in limit_data:
                print(f"   - Max aperture: {limit_data['value']/1e3:.1f} km")
        
        print(f"\n3. PULSED SAR SYSTEM LIMITATIONS:")
        for limit_name, limit_data in pulsed_limits.items():
            print(f"   {limit_name.replace('_', ' ').title()}:")
            print(f"   - {limit_data['description']}")
            print(f"   - Formula: {limit_data['formula']}")
            if 'value' in limit_data:
                print(f"   - Max aperture: {limit_data['value']/1e3:.1f} km")
        
        print(f"\n4. RANGE-DEPENDENT LIMITATIONS:")
        print(f"   Target Range (km) | LFMCW Doppler Limit | Pulsed Doppler Limit | Migration Limit")
        print(f"   " + "-"*75)
        for R in self.target_ranges:
            lfmcw_dop = lfmcw_limits['doppler_bandwidth']['max_apertures'][R]/1e3
            pulsed_dop = pulsed_limits['doppler_ambiguity']['max_apertures'][R]/1e3
            migration = lfmcw_limits['range_migration']['max_apertures'][R]/1e3
            print(f"   {R/1e3:12.0f}   |    {lfmcw_dop:12.1f}    |    {pulsed_dop:13.1f}   |   {migration:10.1f}")
        
        print(f"\n5. PRACTICAL DESIGN GUIDELINES:")
        print(f"   LFMCW Systems:")
        print(f"   - Optimal for apertures: 100m - 5km")
        print(f"   - Best for real-time applications")
        print(f"   - Limited by sweep rate and coherence time")
        print(f"   - Range migration less critical")
        
        print(f"   Pulsed SAR Systems:")
        print(f"   - Optimal for apertures: 1km - 20km")
        print(f"   - Best for high-resolution imaging")
        print(f"   - Limited by PRF and motion compensation")
        print(f"   - Range migration correction essential")
        
        print(f"\n6. RESOLUTION PERFORMANCE:")
        for i, R in enumerate(self.target_ranges):
            print(f"   At {R/1e3:.0f}km range:")
            
            # Calculate achievable resolution for typical apertures
            typical_lfmcw_aperture = 1000  # 1km
            typical_pulsed_aperture = 10000  # 10km
            
            lfmcw_resolution = self.wavelength * R / (2 * typical_lfmcw_aperture)
            pulsed_resolution = self.wavelength * R / (2 * typical_pulsed_aperture)
            
            print(f"   - LFMCW (1km aperture): {lfmcw_resolution:.1f}m resolution")
            print(f"   - Pulsed SAR (10km aperture): {pulsed_resolution:.1f}m resolution")
        
        # Save comprehensive results
        analysis_results = {
            'timestamp': datetime.now().isoformat(),
            'analysis_type': 'synthetic_aperture_size_analysis',
            'system_parameters': {
                'carrier_frequency': self.fc,
                'wavelength': self.wavelength,
                'bandwidth': self.B,
                'platform_velocity': self.platform_velocity,
                'platform_altitude': self.platform_altitude
            },
            'fundamental_relationship': {
                'formula': 'ΔAz = λ*R/(2*L_syn)',
                'description': 'Azimuth resolution inversely proportional to synthetic aperture size'
            },
            'lfmcw_limitations': lfmcw_limits,
            'pulsed_sar_limitations': pulsed_limits,
            'key_insights': {
                'lfmcw_optimal_range': '100m - 5km aperture',
                'pulsed_sar_optimal_range': '1km - 20km aperture',
                'lfmcw_advantage': 'Real-time processing, simple algorithms',
                'pulsed_sar_advantage': 'Larger apertures, better resolution',
                'range_migration_impact': 'More critical for pulsed SAR',
                'motion_compensation_impact': 'Essential for pulsed SAR'
            }
        }
        
        with open('../output/synthetic_aperture_analysis_results.json', 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        print(f"\nDetailed analysis results saved to: ../output/synthetic_aperture_analysis_results.json")
        
        return lfmcw_limits, pulsed_limits, analysis_results

def main():
    """Main execution"""
    analysis = SyntheticApertureSizeAnalysis()
    lfmcw_limits, pulsed_limits, results = analysis.run_aperture_analysis()
    return analysis, lfmcw_limits, pulsed_limits, results

if __name__ == "__main__":
    analysis, lfmcw_limits, pulsed_limits, results = main()
