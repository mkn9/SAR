#!/usr/bin/env python3
"""
Spotlight SAR vs Stripmap SAR Performance Analysis
Comprehensive comparison of two fundamental SAR imaging modes:

1. Stripmap SAR: Antenna beam fixed, continuous ground coverage
2. Spotlight SAR: Antenna beam steered to illuminate same area, higher resolution

Key Performance Parameters Analyzed:
- Azimuth Resolution
- Swath Width  
- Coverage Rate
- Integration Time
- Doppler Bandwidth
- Processing Complexity
- Data Rate Requirements
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append('../../SAR_Benchmark_Validation_2024/code')
from sar_model_final import FinalSARModel
import json
from datetime import datetime

class SpotlightVsStripmapAnalysis:
    def __init__(self, fc=10e9, B=100e6, antenna_length=5.0):
        """Initialize Spotlight vs Stripmap SAR analysis"""
        self.fc = fc  # Carrier frequency
        self.B = B    # Bandwidth
        self.c = 3e8  # Speed of light
        self.wavelength = self.c / fc
        self.antenna_length = antenna_length  # Physical antenna length (m)
        
        # Platform parameters
        self.platform_velocity = 200  # m/s
        self.platform_altitude = 500e3  # 500 km altitude
        
        # System parameters
        self.range_resolution = self.c / (2 * B)
        self.antenna_beamwidth = self.wavelength / antenna_length  # radians
        
        print(f"Spotlight vs Stripmap SAR Analysis")
        print(f"System Parameters:")
        print(f"  Carrier Frequency: {fc/1e9:.1f} GHz")
        print(f"  Wavelength: {self.wavelength*100:.2f} cm")
        print(f"  Bandwidth: {B/1e6:.1f} MHz")
        print(f"  Range Resolution: {self.range_resolution:.3f} m")
        print(f"  Antenna Length: {antenna_length:.1f} m")
        print(f"  Antenna Beamwidth: {np.degrees(self.antenna_beamwidth):.2f}°")
        print(f"  Platform Velocity: {self.platform_velocity} m/s")
        print(f"  Platform Altitude: {self.platform_altitude/1e3:.0f} km")
    
    def calculate_stripmap_performance(self, target_range):
        """Calculate Stripmap SAR performance parameters"""
        print(f"\nStripmap SAR Analysis for range {target_range/1e3:.0f}km:")
        
        # 1. Azimuth Resolution (Stripmap)
        # Limited by physical antenna beamwidth
        # ΔAz = λ*R/(2*La) where La is physical antenna length
        azimuth_resolution = self.wavelength * target_range / (2 * self.antenna_length)
        
        # 2. Swath Width (Stripmap)
        # Determined by pulse timing and range ambiguities
        # For simplicity, assume 50km swath (typical)
        swath_width = 50e3  # meters
        
        # 3. Integration Time (Stripmap)
        # Time for target to cross antenna beam
        beam_footprint_azimuth = self.antenna_beamwidth * target_range
        integration_time = beam_footprint_azimuth / self.platform_velocity
        
        # 4. Synthetic Aperture Length (Stripmap)
        synthetic_aperture_length = self.platform_velocity * integration_time
        
        # 5. Doppler Bandwidth (Stripmap)
        # BD = 2*V/λ for full aperture
        doppler_bandwidth = 2 * self.platform_velocity / self.wavelength
        
        # 6. Coverage Rate (Stripmap)
        # Area covered per unit time
        coverage_rate = swath_width * self.platform_velocity  # m²/s
        
        # 7. Data Rate (Stripmap)
        # Depends on sampling requirements
        PRF = doppler_bandwidth * 1.2  # 20% oversampling
        range_samples = int(2 * swath_width / self.c * self.B * 2)  # Range sampling
        data_rate = PRF * range_samples * 2 * 8  # Complex samples, 8 bits/sample
        
        # 8. Number of Looks (Stripmap)
        # For multi-looking to reduce speckle
        num_looks = max(1, int(synthetic_aperture_length / azimuth_resolution))
        
        stripmap_params = {
            'azimuth_resolution': azimuth_resolution,
            'swath_width': swath_width,
            'integration_time': integration_time,
            'synthetic_aperture_length': synthetic_aperture_length,
            'doppler_bandwidth': doppler_bandwidth,
            'coverage_rate': coverage_rate,
            'data_rate': data_rate,
            'PRF': PRF,
            'num_looks': num_looks,
            'beam_footprint_azimuth': beam_footprint_azimuth
        }
        
        print(f"  Azimuth Resolution: {azimuth_resolution:.2f} m")
        print(f"  Swath Width: {swath_width/1e3:.1f} km")
        print(f"  Integration Time: {integration_time:.3f} s")
        print(f"  Synthetic Aperture: {synthetic_aperture_length:.1f} m")
        print(f"  Doppler Bandwidth: {doppler_bandwidth:.1f} Hz")
        print(f"  Coverage Rate: {coverage_rate/1e6:.1f} km²/s")
        
        return stripmap_params
    
    def calculate_spotlight_performance(self, target_range, spotlight_angle_deg=2.0):
        """Calculate Spotlight SAR performance parameters"""
        print(f"\nSpotlight SAR Analysis for range {target_range/1e3:.0f}km:")
        
        spotlight_angle = np.radians(spotlight_angle_deg)
        
        # 1. Azimuth Resolution (Spotlight)
        # Much better than stripmap due to longer synthetic aperture
        # ΔAz = λ*R/(2*L_syn) where L_syn is much larger than physical antenna
        
        # Maximum synthetic aperture limited by steering angle
        max_synthetic_aperture = 2 * target_range * np.tan(spotlight_angle / 2)
        
        # Practical limit: coherence time and processing constraints
        coherence_time = 10.0  # seconds (longer for spotlight due to controlled geometry)
        max_aperture_coherence = self.platform_velocity * coherence_time
        
        # Use the smaller of the two limits
        synthetic_aperture_length = min(max_synthetic_aperture, max_aperture_coherence)
        
        azimuth_resolution = self.wavelength * target_range / (2 * synthetic_aperture_length)
        
        # 2. Swath Width (Spotlight)
        # Much smaller than stripmap - focused on specific area
        # Limited by beam steering capability and scene size
        scene_size = 2 * target_range * np.tan(spotlight_angle / 2)  # Scene diameter
        swath_width = scene_size  # Effective swath for spotlight
        
        # 3. Integration Time (Spotlight)
        # Time to collect full synthetic aperture while steering beam
        integration_time = synthetic_aperture_length / self.platform_velocity
        
        # 4. Doppler Bandwidth (Spotlight)
        # Larger than stripmap due to beam steering
        # BD = 2*V*sin(θ_max)/λ where θ_max is maximum steering angle
        doppler_bandwidth = 2 * self.platform_velocity * np.sin(spotlight_angle/2) / self.wavelength
        
        # 5. Coverage Rate (Spotlight)
        # Much lower than stripmap due to dwelling on specific areas
        area_per_scene = np.pi * (scene_size/2)**2  # Circular scene area
        time_per_scene = integration_time
        coverage_rate = area_per_scene / time_per_scene
        
        # 6. Data Rate (Spotlight)
        # Higher PRF needed for larger Doppler bandwidth
        PRF = doppler_bandwidth * 1.5  # 50% oversampling for spotlight
        range_samples = int(2 * scene_size / self.c * self.B * 2)
        data_rate = PRF * range_samples * 2 * 8  # Complex samples
        
        # 7. Number of Looks (Spotlight)
        # More looks available due to longer aperture
        num_looks = max(1, int(synthetic_aperture_length / azimuth_resolution))
        
        # 8. Beam Steering Requirements
        angular_velocity = spotlight_angle / integration_time  # rad/s
        beam_steering_rate = np.degrees(angular_velocity)  # deg/s
        
        spotlight_params = {
            'azimuth_resolution': azimuth_resolution,
            'swath_width': swath_width,
            'integration_time': integration_time,
            'synthetic_aperture_length': synthetic_aperture_length,
            'doppler_bandwidth': doppler_bandwidth,
            'coverage_rate': coverage_rate,
            'data_rate': data_rate,
            'PRF': PRF,
            'num_looks': num_looks,
            'scene_size': scene_size,
            'beam_steering_rate': beam_steering_rate,
            'spotlight_angle': spotlight_angle_deg
        }
        
        print(f"  Azimuth Resolution: {azimuth_resolution:.2f} m")
        print(f"  Scene Size: {scene_size/1e3:.1f} km")
        print(f"  Integration Time: {integration_time:.1f} s")
        print(f"  Synthetic Aperture: {synthetic_aperture_length:.1f} m")
        print(f"  Doppler Bandwidth: {doppler_bandwidth:.1f} Hz")
        print(f"  Coverage Rate: {coverage_rate/1e6:.1f} km²/s")
        print(f"  Beam Steering Rate: {beam_steering_rate:.2f} deg/s")
        
        return spotlight_params
    
    def calculate_performance_vs_range(self):
        """Calculate performance parameters vs range for both modes"""
        ranges = np.linspace(100e3, 1000e3, 20)  # 100km to 1000km
        
        stripmap_results = {
            'ranges': ranges,
            'azimuth_resolution': [],
            'swath_width': [],
            'coverage_rate': [],
            'integration_time': [],
            'synthetic_aperture': [],
            'doppler_bandwidth': []
        }
        
        spotlight_results = {
            'ranges': ranges,
            'azimuth_resolution': [],
            'swath_width': [],
            'coverage_rate': [],
            'integration_time': [],
            'synthetic_aperture': [],
            'doppler_bandwidth': []
        }
        
        for R in ranges:
            # Stripmap calculations
            strip_params = self.calculate_stripmap_performance(R)
            stripmap_results['azimuth_resolution'].append(strip_params['azimuth_resolution'])
            stripmap_results['swath_width'].append(strip_params['swath_width'])
            stripmap_results['coverage_rate'].append(strip_params['coverage_rate'])
            stripmap_results['integration_time'].append(strip_params['integration_time'])
            stripmap_results['synthetic_aperture'].append(strip_params['synthetic_aperture_length'])
            stripmap_results['doppler_bandwidth'].append(strip_params['doppler_bandwidth'])
            
            # Spotlight calculations
            spot_params = self.calculate_spotlight_performance(R)
            spotlight_results['azimuth_resolution'].append(spot_params['azimuth_resolution'])
            spotlight_results['swath_width'].append(spot_params['swath_width'])
            spotlight_results['coverage_rate'].append(spot_params['coverage_rate'])
            spotlight_results['integration_time'].append(spot_params['integration_time'])
            spotlight_results['synthetic_aperture'].append(spot_params['synthetic_aperture_length'])
            spotlight_results['doppler_bandwidth'].append(spot_params['doppler_bandwidth'])
        
        return stripmap_results, spotlight_results
    
    def create_comprehensive_comparison_plots(self, stripmap_results, spotlight_results):
        """Create comprehensive comparison plots"""
        fig = plt.figure(figsize=(20, 16))
        
        ranges_km = np.array(stripmap_results['ranges']) / 1e3
        
        # Plot 1: Azimuth Resolution vs Range
        ax1 = plt.subplot(2, 4, 1)
        ax1.plot(ranges_km, np.array(stripmap_results['azimuth_resolution']), 
                'b-o', linewidth=2, markersize=6, label='Stripmap SAR')
        ax1.plot(ranges_km, np.array(spotlight_results['azimuth_resolution']), 
                'r-s', linewidth=2, markersize=6, label='Spotlight SAR')
        
        ax1.set_xlabel('Range (km)', fontsize=12)
        ax1.set_ylabel('Azimuth Resolution (m)', fontsize=12)
        ax1.set_title('Azimuth Resolution vs Range\nΔAz = λR/(2L_syn)', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_yscale('log')
        
        # Plot 2: Swath Width vs Range
        ax2 = plt.subplot(2, 4, 2)
        ax2.plot(ranges_km, np.array(stripmap_results['swath_width'])/1e3, 
                'b-o', linewidth=2, markersize=6, label='Stripmap SAR')
        ax2.plot(ranges_km, np.array(spotlight_results['swath_width'])/1e3, 
                'r-s', linewidth=2, markersize=6, label='Spotlight SAR')
        
        ax2.set_xlabel('Range (km)', fontsize=12)
        ax2.set_ylabel('Swath Width (km)', fontsize=12)
        ax2.set_title('Swath Width vs Range', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: Coverage Rate vs Range
        ax3 = plt.subplot(2, 4, 3)
        ax3.plot(ranges_km, np.array(stripmap_results['coverage_rate'])/1e6, 
                'b-o', linewidth=2, markersize=6, label='Stripmap SAR')
        ax3.plot(ranges_km, np.array(spotlight_results['coverage_rate'])/1e6, 
                'r-s', linewidth=2, markersize=6, label='Spotlight SAR')
        
        ax3.set_xlabel('Range (km)', fontsize=12)
        ax3.set_ylabel('Coverage Rate (km²/s)', fontsize=12)
        ax3.set_title('Coverage Rate vs Range', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.set_yscale('log')
        
        # Plot 4: Integration Time vs Range
        ax4 = plt.subplot(2, 4, 4)
        ax4.plot(ranges_km, np.array(stripmap_results['integration_time']), 
                'b-o', linewidth=2, markersize=6, label='Stripmap SAR')
        ax4.plot(ranges_km, np.array(spotlight_results['integration_time']), 
                'r-s', linewidth=2, markersize=6, label='Spotlight SAR')
        
        ax4.set_xlabel('Range (km)', fontsize=12)
        ax4.set_ylabel('Integration Time (s)', fontsize=12)
        ax4.set_title('Integration Time vs Range', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        ax4.set_yscale('log')
        
        # Plot 5: Synthetic Aperture Length vs Range
        ax5 = plt.subplot(2, 4, 5)
        ax5.plot(ranges_km, np.array(stripmap_results['synthetic_aperture'])/1e3, 
                'b-o', linewidth=2, markersize=6, label='Stripmap SAR')
        ax5.plot(ranges_km, np.array(spotlight_results['synthetic_aperture'])/1e3, 
                'r-s', linewidth=2, markersize=6, label='Spotlight SAR')
        
        ax5.set_xlabel('Range (km)', fontsize=12)
        ax5.set_ylabel('Synthetic Aperture (km)', fontsize=12)
        ax5.set_title('Synthetic Aperture Length vs Range', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        
        # Plot 6: Performance Trade-off Analysis
        ax6 = plt.subplot(2, 4, 6)
        
        # Calculate resolution improvement factor
        resolution_improvement = np.array(stripmap_results['azimuth_resolution']) / np.array(spotlight_results['azimuth_resolution'])
        coverage_ratio = np.array(stripmap_results['coverage_rate']) / np.array(spotlight_results['coverage_rate'])
        
        ax6.plot(ranges_km, resolution_improvement, 'g-o', linewidth=2, markersize=6, 
                label='Resolution Improvement\n(Spotlight/Stripmap)')
        ax6_twin = ax6.twinx()
        ax6_twin.plot(ranges_km, coverage_ratio, 'orange', linestyle='-', marker='s', 
                     linewidth=2, markersize=6, label='Coverage Ratio\n(Stripmap/Spotlight)')
        
        ax6.set_xlabel('Range (km)', fontsize=12)
        ax6.set_ylabel('Resolution Improvement Factor', fontsize=12, color='green')
        ax6_twin.set_ylabel('Coverage Rate Ratio', fontsize=12, color='orange')
        ax6.set_title('Performance Trade-offs', fontweight='bold')
        ax6.grid(True, alpha=0.3)
        ax6.legend(loc='upper left')
        ax6_twin.legend(loc='upper right')
        ax6.set_yscale('log')
        ax6_twin.set_yscale('log')
        
        # Plot 7: Mode Comparison Table
        ax7 = plt.subplot(2, 4, 7)
        
        # Use middle range (500km) for comparison
        mid_idx = len(ranges_km) // 2
        
        comparison_data = [
            ['Parameter', 'Stripmap', 'Spotlight', 'Advantage'],
            ['Azimuth Resolution (m)', 
             f'{stripmap_results["azimuth_resolution"][mid_idx]:.1f}',
             f'{spotlight_results["azimuth_resolution"][mid_idx]:.2f}',
             'Spotlight'],
            ['Swath Width (km)', 
             f'{stripmap_results["swath_width"][mid_idx]/1e3:.1f}',
             f'{spotlight_results["swath_width"][mid_idx]/1e3:.1f}',
             'Stripmap'],
            ['Coverage Rate (km²/s)', 
             f'{stripmap_results["coverage_rate"][mid_idx]/1e6:.1f}',
             f'{spotlight_results["coverage_rate"][mid_idx]/1e6:.3f}',
             'Stripmap'],
            ['Integration Time (s)', 
             f'{stripmap_results["integration_time"][mid_idx]:.2f}',
             f'{spotlight_results["integration_time"][mid_idx]:.1f}',
             'Stripmap'],
            ['Synthetic Aperture (km)', 
             f'{stripmap_results["synthetic_aperture"][mid_idx]/1e3:.1f}',
             f'{spotlight_results["synthetic_aperture"][mid_idx]/1e3:.1f}',
             'Spotlight'],
            ['Processing Complexity', 'Medium', 'High', 'Stripmap'],
            ['Data Rate', 'Medium', 'High', 'Stripmap'],
            ['Applications', 'Wide Area\nSurveillance', 'High-Res\nImaging', 'Mission\nDependent']
        ]
        
        table = ax7.table(cellText=comparison_data[1:], colLabels=comparison_data[0],
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        
        # Color code advantages
        for i in range(1, len(comparison_data)):
            advantage_cell = table[(i, 3)]
            if 'Spotlight' in comparison_data[i][3]:
                advantage_cell.set_facecolor('#FFE6E6')  # Light red
            elif 'Stripmap' in comparison_data[i][3]:
                advantage_cell.set_facecolor('#E6F3FF')  # Light blue
            else:
                advantage_cell.set_facecolor('#F0F0F0')  # Light gray
        
        # Header styling
        for j in range(4):
            table[(0, j)].set_facecolor('#4CAF50')
            table[(0, j)].set_text_props(weight='bold', color='white')
        
        ax7.set_title('Mode Comparison\n(at 500km range)', fontweight='bold')
        ax7.axis('off')
        
        # Plot 8: Operational Concepts Diagram
        ax8 = plt.subplot(2, 4, 8)
        
        # Create conceptual diagram
        ax8.text(0.5, 0.9, 'SAR Mode Concepts', ha='center', va='center', 
                transform=ax8.transAxes, fontsize=14, fontweight='bold')
        
        # Stripmap concept
        ax8.text(0.25, 0.75, 'STRIPMAP SAR', ha='center', va='center',
                transform=ax8.transAxes, fontsize=12, fontweight='bold', color='blue')
        ax8.text(0.25, 0.65, '• Fixed beam pointing', ha='center', va='center',
                transform=ax8.transAxes, fontsize=10)
        ax8.text(0.25, 0.60, '• Continuous strip coverage', ha='center', va='center',
                transform=ax8.transAxes, fontsize=10)
        ax8.text(0.25, 0.55, '• Wide swath', ha='center', va='center',
                transform=ax8.transAxes, fontsize=10)
        ax8.text(0.25, 0.50, '• Moderate resolution', ha='center', va='center',
                transform=ax8.transAxes, fontsize=10)
        ax8.text(0.25, 0.45, '• High coverage rate', ha='center', va='center',
                transform=ax8.transAxes, fontsize=10)
        
        # Spotlight concept
        ax8.text(0.75, 0.75, 'SPOTLIGHT SAR', ha='center', va='center',
                transform=ax8.transAxes, fontsize=12, fontweight='bold', color='red')
        ax8.text(0.75, 0.65, '• Steered beam pointing', ha='center', va='center',
                transform=ax8.transAxes, fontsize=10)
        ax8.text(0.75, 0.60, '• Focused area illumination', ha='center', va='center',
                transform=ax8.transAxes, fontsize=10)
        ax8.text(0.75, 0.55, '• Small scene size', ha='center', va='center',
                transform=ax8.transAxes, fontsize=10)
        ax8.text(0.75, 0.50, '• High resolution', ha='center', va='center',
                transform=ax8.transAxes, fontsize=10)
        ax8.text(0.75, 0.45, '• Low coverage rate', ha='center', va='center',
                transform=ax8.transAxes, fontsize=10)
        
        # Key equations
        ax8.text(0.5, 0.35, 'Key Equations:', ha='center', va='center',
                transform=ax8.transAxes, fontsize=12, fontweight='bold')
        ax8.text(0.5, 0.25, 'Azimuth Resolution: ΔAz = λR/(2L_syn)', ha='center', va='center',
                transform=ax8.transAxes, fontsize=10)
        ax8.text(0.5, 0.20, 'Stripmap: L_syn = λR/θ_ant (antenna limited)', ha='center', va='center',
                transform=ax8.transAxes, fontsize=10)
        ax8.text(0.5, 0.15, 'Spotlight: L_syn >> λR/θ_ant (steering limited)', ha='center', va='center',
                transform=ax8.transAxes, fontsize=10)
        ax8.text(0.5, 0.05, 'Trade-off: Resolution ↔ Coverage', ha='center', va='center',
                transform=ax8.transAxes, fontsize=12, fontweight='bold', color='purple')
        
        ax8.set_xlim(0, 1)
        ax8.set_ylim(0, 1)
        ax8.axis('off')
        
        plt.suptitle('Spotlight SAR vs Stripmap SAR Performance Analysis\nComprehensive Comparison of Key Parameters', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('../output/spotlight_vs_stripmap_sar_analysis.png', dpi=300, bbox_inches='tight')
        print("\nSpotlight vs Stripmap SAR analysis plots saved to: ../output/spotlight_vs_stripmap_sar_analysis.png")
        plt.close()
    
    def run_comprehensive_analysis(self):
        """Run comprehensive Spotlight vs Stripmap analysis"""
        print("SPOTLIGHT SAR vs STRIPMAP SAR PERFORMANCE ANALYSIS")
        print("="*80)
        
        # Create output directory
        os.makedirs('../output', exist_ok=True)
        
        # Calculate performance vs range for both modes
        stripmap_results, spotlight_results = self.calculate_performance_vs_range()
        
        # Create comprehensive comparison plots
        self.create_comprehensive_comparison_plots(stripmap_results, spotlight_results)
        
        # Detailed analysis summary
        print(f"\n" + "="*80)
        print("SPOTLIGHT vs STRIPMAP SAR ANALYSIS SUMMARY")
        print("="*80)
        
        print(f"\n1. FUNDAMENTAL DIFFERENCES:")
        print(f"   Stripmap SAR:")
        print(f"   - Antenna beam fixed relative to platform")
        print(f"   - Continuous ground coverage along track")
        print(f"   - Synthetic aperture limited by antenna beamwidth")
        print(f"   - Formula: L_syn = λR/θ_antenna = {self.antenna_length:.1f}m (physical antenna)")
        
        print(f"\n   Spotlight SAR:")
        print(f"   - Antenna beam electronically/mechanically steered")
        print(f"   - Focused illumination of specific areas")
        print(f"   - Synthetic aperture limited by steering capability")
        print(f"   - Formula: L_syn = 2R*tan(θ_steer/2) (steering angle limited)")
        
        print(f"\n2. PERFORMANCE AT 500KM RANGE:")
        mid_idx = len(stripmap_results['ranges']) // 2
        
        print(f"   Stripmap SAR:")
        print(f"   - Azimuth Resolution: {stripmap_results['azimuth_resolution'][mid_idx]:.1f} m")
        print(f"   - Swath Width: {stripmap_results['swath_width'][mid_idx]/1e3:.1f} km")
        print(f"   - Coverage Rate: {stripmap_results['coverage_rate'][mid_idx]/1e6:.1f} km²/s")
        print(f"   - Integration Time: {stripmap_results['integration_time'][mid_idx]:.2f} s")
        print(f"   - Synthetic Aperture: {stripmap_results['synthetic_aperture'][mid_idx]/1e3:.1f} km")
        
        print(f"\n   Spotlight SAR:")
        print(f"   - Azimuth Resolution: {spotlight_results['azimuth_resolution'][mid_idx]:.2f} m")
        print(f"   - Scene Size: {spotlight_results['swath_width'][mid_idx]/1e3:.1f} km")
        print(f"   - Coverage Rate: {spotlight_results['coverage_rate'][mid_idx]/1e6:.3f} km²/s")
        print(f"   - Integration Time: {spotlight_results['integration_time'][mid_idx]:.1f} s")
        print(f"   - Synthetic Aperture: {spotlight_results['synthetic_aperture'][mid_idx]/1e3:.1f} km")
        
        # Calculate improvement factors
        resolution_improvement = stripmap_results['azimuth_resolution'][mid_idx] / spotlight_results['azimuth_resolution'][mid_idx]
        coverage_ratio = stripmap_results['coverage_rate'][mid_idx] / spotlight_results['coverage_rate'][mid_idx]
        
        print(f"\n3. PERFORMANCE TRADE-OFFS:")
        print(f"   Resolution Improvement (Spotlight): {resolution_improvement:.1f}x better")
        print(f"   Coverage Rate Advantage (Stripmap): {coverage_ratio:.0f}x higher")
        print(f"   Integration Time Ratio: {spotlight_results['integration_time'][mid_idx]/stripmap_results['integration_time'][mid_idx]:.1f}x longer")
        
        print(f"\n4. KEY CALCULATIONS EXPLAINED:")
        print(f"   Stripmap Azimuth Resolution:")
        print(f"   - ΔAz = λ*R/(2*La) where La = physical antenna length")
        print(f"   - Limited by {self.antenna_length:.1f}m antenna → {stripmap_results['azimuth_resolution'][mid_idx]:.1f}m resolution")
        print(f"   - Synthetic aperture = beam footprint = {stripmap_results['synthetic_aperture'][mid_idx]/1e3:.1f}km")
        
        print(f"\n   Spotlight Azimuth Resolution:")
        print(f"   - ΔAz = λ*R/(2*L_syn) where L_syn >> La")
        print(f"   - L_syn limited by steering angle and coherence time")
        print(f"   - Achieves {spotlight_results['azimuth_resolution'][mid_idx]:.2f}m resolution with {spotlight_results['synthetic_aperture'][mid_idx]/1e3:.1f}km aperture")
        
        print(f"\n   Coverage Rate Calculations:")
        print(f"   - Stripmap: Swath_width × Velocity = {stripmap_results['swath_width'][mid_idx]/1e3:.1f}km × {self.platform_velocity}m/s")
        print(f"   - Spotlight: Scene_area / Integration_time = circular area / dwell time")
        
        print(f"\n5. SYSTEM DESIGN IMPLICATIONS:")
        print(f"   Stripmap Requirements:")
        print(f"   - Fixed antenna pointing (simpler)")
        print(f"   - Lower PRF requirements")
        print(f"   - Continuous data collection")
        print(f"   - Simpler processing algorithms")
        
        print(f"   Spotlight Requirements:")
        print(f"   - Beam steering capability (complex)")
        print(f"   - Higher PRF for Doppler bandwidth")
        print(f"   - Intermittent data collection")
        print(f"   - Complex motion compensation")
        print(f"   - Advanced focusing algorithms")
        
        print(f"\n6. APPLICATION SUITABILITY:")
        print(f"   Stripmap SAR:")
        print(f"   - Wide area surveillance")
        print(f"   - Mapping and cartography")
        print(f"   - Environmental monitoring")
        print(f"   - Disaster response (large area coverage)")
        
        print(f"   Spotlight SAR:")
        print(f"   - High-resolution target identification")
        print(f"   - Detailed infrastructure analysis")
        print(f"   - Military reconnaissance")
        print(f"   - Scientific studies requiring fine detail")
        
        # Save comprehensive results
        analysis_results = {
            'timestamp': datetime.now().isoformat(),
            'analysis_type': 'spotlight_vs_stripmap_sar',
            'system_parameters': {
                'carrier_frequency': self.fc,
                'wavelength': self.wavelength,
                'bandwidth': self.B,
                'antenna_length': self.antenna_length,
                'antenna_beamwidth_deg': np.degrees(self.antenna_beamwidth),
                'platform_velocity': self.platform_velocity,
                'platform_altitude': self.platform_altitude
            },
            'performance_comparison_500km': {
                'stripmap': {
                    'azimuth_resolution_m': stripmap_results['azimuth_resolution'][mid_idx],
                    'swath_width_km': stripmap_results['swath_width'][mid_idx]/1e3,
                    'coverage_rate_km2_per_s': stripmap_results['coverage_rate'][mid_idx]/1e6,
                    'integration_time_s': stripmap_results['integration_time'][mid_idx],
                    'synthetic_aperture_km': stripmap_results['synthetic_aperture'][mid_idx]/1e3
                },
                'spotlight': {
                    'azimuth_resolution_m': spotlight_results['azimuth_resolution'][mid_idx],
                    'scene_size_km': spotlight_results['swath_width'][mid_idx]/1e3,
                    'coverage_rate_km2_per_s': spotlight_results['coverage_rate'][mid_idx]/1e6,
                    'integration_time_s': spotlight_results['integration_time'][mid_idx],
                    'synthetic_aperture_km': spotlight_results['synthetic_aperture'][mid_idx]/1e3
                }
            },
            'key_trade_offs': {
                'resolution_improvement_factor': resolution_improvement,
                'coverage_rate_ratio': coverage_ratio,
                'integration_time_ratio': spotlight_results['integration_time'][mid_idx]/stripmap_results['integration_time'][mid_idx]
            },
            'fundamental_equations': {
                'azimuth_resolution': 'ΔAz = λ*R/(2*L_syn)',
                'stripmap_aperture': 'L_syn = λ*R/θ_antenna (antenna limited)',
                'spotlight_aperture': 'L_syn = 2*R*tan(θ_steer/2) (steering limited)',
                'coverage_rate_stripmap': 'Coverage = Swath_width × Velocity',
                'coverage_rate_spotlight': 'Coverage = Scene_area / Integration_time'
            }
        }
        
        with open('../output/spotlight_vs_stripmap_results.json', 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        print(f"\nDetailed analysis results saved to: ../output/spotlight_vs_stripmap_results.json")
        
        return stripmap_results, spotlight_results, analysis_results

def main():
    """Main execution"""
    analysis = SpotlightVsStripmapAnalysis()
    stripmap_results, spotlight_results, results = analysis.run_comprehensive_analysis()
    return analysis, stripmap_results, spotlight_results, results

if __name__ == "__main__":
    analysis, stripmap_results, spotlight_results, results = main()
