#!/usr/bin/env python3
"""
LFMCW vs Pulsed SAR Analysis: Range-Doppler Map (RDM) Processing
Detailed comparison of Linear Frequency Modulated Continuous Wave radar 
with Range-Doppler Maps versus traditional pulsed SAR with pulse compression

Key Concepts:
1. LFMCW: Continuous transmission with frequency sweeps
2. Pulsed SAR: Discrete pulses with matched filtering
3. RDM: 2D processing in range-Doppler domain
4. Pulse Compression: Time-domain matched filtering
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append('../../SAR_Benchmark_Validation_2024/code')
from sar_model_final import FinalSARModel
from scipy.signal import chirp, hilbert, spectrogram
from scipy.fft import fft, ifft, fftfreq, fftshift
import json
from datetime import datetime

class LFMCWvsPulsedSARAnalysis:
    def __init__(self, fc=10e9, B=100e6, T_sweep=1e-3, PRF=1000):
        """Initialize LFMCW vs Pulsed SAR comparison"""
        self.fc = fc  # Carrier frequency
        self.B = B    # Bandwidth
        self.T_sweep = T_sweep  # LFMCW sweep time
        self.PRF = PRF  # Pulse repetition frequency for pulsed system
        self.c = 3e8  # Speed of light
        
        # Derived parameters
        self.wavelength = self.c / fc
        self.range_resolution = self.c / (2 * B)
        self.max_range = self.c * T_sweep / 2  # For LFMCW
        
        # Platform parameters
        self.platform_velocity = 200  # m/s
        self.integration_time = 1.0   # seconds
        
        print(f"LFMCW vs Pulsed SAR Analysis")
        print(f"System Parameters:")
        print(f"  Carrier Frequency: {fc/1e9:.1f} GHz")
        print(f"  Bandwidth: {B/1e6:.1f} MHz")
        print(f"  Range Resolution: {self.range_resolution:.3f} m")
        print(f"  LFMCW Sweep Time: {T_sweep*1e3:.1f} ms")
        print(f"  Max Range (LFMCW): {self.max_range/1e3:.1f} km")
        print(f"  PRF (Pulsed): {PRF} Hz")
    
    def generate_lfmcw_signal(self, targets, fs=200e6, num_sweeps=128):
        """Generate LFMCW signal with multiple targets"""
        print(f"\nGenerating LFMCW Signal:")
        print(f"  Targets: {len(targets)}")
        print(f"  Sweeps: {num_sweeps}")
        print(f"  Sampling Rate: {fs/1e6:.1f} MHz")
        
        # Time vector for one sweep
        samples_per_sweep = int(self.T_sweep * fs)
        t_sweep = np.linspace(0, self.T_sweep, samples_per_sweep)
        
        # Generate LFM chirp for one sweep
        chirp_rate = self.B / self.T_sweep
        reference_chirp = chirp(t_sweep, self.fc - self.B/2, self.T_sweep, 
                               self.fc + self.B/2, method='linear')
        
        # Initialize signal matrix [range_samples, doppler_samples]
        received_signal = np.zeros((samples_per_sweep, num_sweeps), dtype=complex)
        
        # Simulate target returns for each sweep
        for sweep_idx in range(num_sweeps):
            # Platform position for this sweep
            platform_time = sweep_idx / self.PRF
            platform_position = self.platform_velocity * platform_time
            
            sweep_signal = np.zeros(samples_per_sweep, dtype=complex)
            
            for target in targets:
                target_range = target['range']
                target_azimuth = target.get('azimuth', 0.0)
                target_rcs = target['rcs']
                
                # Calculate instantaneous range considering platform motion
                relative_azimuth = platform_position - target_azimuth
                instantaneous_range = np.sqrt(target_range**2 + relative_azimuth**2)
                
                # Time delay for this range
                time_delay = 2 * instantaneous_range / self.c
                
                # Doppler frequency due to platform motion
                doppler_freq = 2 * self.platform_velocity * relative_azimuth / (self.wavelength * instantaneous_range)
                
                if time_delay < self.T_sweep:
                    # Delayed and Doppler-shifted chirp
                    delay_samples = int(time_delay * fs)
                    if delay_samples < samples_per_sweep:
                        # Create delayed chirp
                        delayed_chirp = np.zeros(samples_per_sweep, dtype=complex)
                        valid_samples = samples_per_sweep - delay_samples
                        
                        # Generate chirp with Doppler shift
                        t_delayed = t_sweep[:valid_samples]
                        doppler_chirp = chirp(t_delayed, 
                                            self.fc - self.B/2 + doppler_freq, 
                                            valid_samples/fs, 
                                            self.fc + self.B/2 + doppler_freq, 
                                            method='linear')
                        
                        delayed_chirp[delay_samples:] = doppler_chirp
                        
                        # Add to sweep signal with appropriate amplitude
                        range_attenuation = (4 * np.pi * instantaneous_range**2)
                        amplitude = np.sqrt(target_rcs) / range_attenuation
                        sweep_signal += amplitude * delayed_chirp
            
            received_signal[:, sweep_idx] = sweep_signal
        
        return received_signal, reference_chirp, t_sweep
    
    def process_lfmcw_rdm(self, received_signal, reference_chirp):
        """Process LFMCW signal to create Range-Doppler Map"""
        print("  Processing LFMCW Range-Doppler Map...")
        
        range_samples, doppler_samples = received_signal.shape
        
        # Range compression: correlate each sweep with reference chirp
        range_compressed = np.zeros_like(received_signal)
        
        # Create reference chirp spectrum for efficient correlation
        ref_spectrum = np.fft.fft(reference_chirp, range_samples)
        ref_spectrum_conj = np.conj(ref_spectrum)
        
        for sweep_idx in range(doppler_samples):
            # FFT of received signal
            received_spectrum = np.fft.fft(received_signal[:, sweep_idx])
            
            # Matched filtering in frequency domain
            compressed_spectrum = received_spectrum * ref_spectrum_conj
            
            # IFFT to get range-compressed signal
            range_compressed[:, sweep_idx] = np.fft.ifft(compressed_spectrum)
        
        # Doppler processing: FFT across sweeps for each range bin
        rdm = np.zeros_like(range_compressed)
        for range_idx in range(range_samples):
            rdm[range_idx, :] = np.fft.fft(range_compressed[range_idx, :])
        
        # Shift zero frequency to center
        rdm = np.fft.fftshift(rdm, axes=1)
        
        return rdm, range_compressed
    
    def generate_pulsed_sar_signal(self, targets, fs=200e6, num_pulses=128):
        """Generate traditional pulsed SAR signal for comparison"""
        print(f"\nGenerating Pulsed SAR Signal:")
        print(f"  Targets: {len(targets)}")
        print(f"  Pulses: {num_pulses}")
        
        # Initialize our existing SAR model
        sar = FinalSARModel(fc=self.fc, B=self.B, Tp=10e-6)
        
        # Generate pulse responses
        pulse_responses = []
        platform_positions = []
        
        for pulse_idx in range(num_pulses):
            # Platform position for this pulse
            platform_time = pulse_idx / self.PRF
            platform_position = self.platform_velocity * platform_time
            platform_positions.append(platform_position)
            
            # Combined response for all targets at this platform position
            combined_response = None
            
            for target in targets:
                target_range = target['range']
                target_azimuth = target.get('azimuth', 0.0)
                target_rcs = target['rcs']
                
                # Calculate instantaneous range
                relative_azimuth = platform_position - target_azimuth
                instantaneous_range = np.sqrt(target_range**2 + relative_azimuth**2)
                
                # Generate point target response
                t, response = sar.point_target_response(R0=instantaneous_range, fs=fs, plot=False)
                
                # Scale by RCS
                response *= target_rcs
                
                # Combine responses
                if combined_response is None:
                    combined_response = response.copy()
                else:
                    min_len = min(len(combined_response), len(response))
                    combined_response[:min_len] += response[:min_len]
            
            pulse_responses.append(combined_response)
        
        return pulse_responses, platform_positions, sar
    
    def process_pulsed_sar(self, pulse_responses, sar):
        """Process pulsed SAR data with traditional pulse compression"""
        print("  Processing Pulsed SAR with Pulse Compression...")
        
        # Range compression for each pulse
        compressed_pulses = []
        for response in pulse_responses:
            compressed = sar.range_compression(response, plot=False)
            compressed_pulses.append(compressed)
        
        # Create 2D SAR data matrix
        max_len = max(len(pulse) for pulse in compressed_pulses)
        sar_data = np.zeros((max_len, len(compressed_pulses)), dtype=complex)
        
        for i, pulse in enumerate(compressed_pulses):
            sar_data[:len(pulse), i] = pulse
        
        # Simple azimuth compression (coherent integration)
        azimuth_compressed = np.zeros_like(sar_data)
        for range_idx in range(sar_data.shape[0]):
            azimuth_compressed[range_idx, :] = np.fft.fft(sar_data[range_idx, :])
        
        return azimuth_compressed, sar_data
    
    def create_comparison_plots(self, lfmcw_rdm, lfmcw_range_compressed, 
                               pulsed_sar_processed, pulsed_sar_raw, targets):
        """Create comprehensive comparison plots"""
        fig = plt.figure(figsize=(20, 16))
        
        # Plot 1: LFMCW Range-Doppler Map
        ax1 = plt.subplot(2, 4, 1)
        rdm_magnitude = np.abs(lfmcw_rdm)
        rdm_db = 20 * np.log10(rdm_magnitude + 1e-10)
        
        range_samples, doppler_samples = rdm_magnitude.shape
        range_axis = np.linspace(0, self.max_range, range_samples)
        doppler_axis = np.linspace(-self.PRF/2, self.PRF/2, doppler_samples)
        
        im1 = ax1.imshow(rdm_db, aspect='auto', origin='lower', 
                        extent=[doppler_axis[0], doppler_axis[-1], 
                               range_axis[0], range_axis[-1]], 
                        cmap='jet')
        ax1.set_xlabel('Doppler Frequency (Hz)')
        ax1.set_ylabel('Range (m)')
        ax1.set_title('LFMCW Range-Doppler Map\n(RDM Processing)', fontweight='bold')
        plt.colorbar(im1, ax=ax1, label='Magnitude (dB)')
        
        # Mark expected targets
        for i, target in enumerate(targets):
            ax1.plot(0, target['range'], 'wo', markersize=10, markeredgecolor='red', 
                    linewidth=2, label=f'Target {i+1}' if i < 2 else '')
        if len(targets) <= 2:
            ax1.legend()
        
        # Plot 2: LFMCW Range Profile (summed across Doppler)
        ax2 = plt.subplot(2, 4, 2)
        range_profile_lfmcw = np.sum(rdm_magnitude, axis=1)
        ax2.plot(range_axis, range_profile_lfmcw, 'b-', linewidth=2, label='LFMCW Range Profile')
        
        for target in targets:
            ax2.axvline(target['range'], color='red', linestyle='--', alpha=0.8, linewidth=2)
        
        ax2.set_xlabel('Range (m)')
        ax2.set_ylabel('Magnitude')
        ax2.set_title('LFMCW Range Profile\n(Doppler Integrated)', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: Pulsed SAR Processed Image
        ax3 = plt.subplot(2, 4, 3)
        sar_magnitude = np.abs(pulsed_sar_processed)
        sar_db = 20 * np.log10(sar_magnitude + 1e-10)
        
        # Create time axis for pulsed SAR
        time_samples = sar_magnitude.shape[0]
        pulse_samples = sar_magnitude.shape[1]
        time_axis = np.linspace(0, time_samples / 200e6 * self.c / 2, time_samples)  # Convert to range
        pulse_axis = np.arange(pulse_samples)
        
        im3 = ax3.imshow(sar_db, aspect='auto', origin='lower',
                        extent=[pulse_axis[0], pulse_axis[-1], 
                               time_axis[0], time_axis[-1]], 
                        cmap='jet')
        ax3.set_xlabel('Pulse Number')
        ax3.set_ylabel('Range (m)')
        ax3.set_title('Pulsed SAR Processed\n(Pulse Compression)', fontweight='bold')
        plt.colorbar(im3, ax=ax3, label='Magnitude (dB)')
        
        # Mark expected targets
        for i, target in enumerate(targets):
            ax3.plot(pulse_samples//2, target['range'], 'wo', markersize=10, 
                    markeredgecolor='red', linewidth=2)
        
        # Plot 4: Pulsed SAR Range Profile
        ax4 = plt.subplot(2, 4, 4)
        range_profile_sar = np.sum(sar_magnitude, axis=1)
        ax4.plot(time_axis, range_profile_sar, 'g-', linewidth=2, label='Pulsed SAR Range Profile')
        
        for target in targets:
            ax4.axvline(target['range'], color='red', linestyle='--', alpha=0.8, linewidth=2)
        
        ax4.set_xlabel('Range (m)')
        ax4.set_ylabel('Magnitude')
        ax4.set_title('Pulsed SAR Range Profile\n(Azimuth Integrated)', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # Plot 5: Direct Range Profile Comparison
        ax5 = plt.subplot(2, 4, 5)
        
        # Normalize profiles for comparison
        range_profile_lfmcw_norm = range_profile_lfmcw / np.max(range_profile_lfmcw)
        range_profile_sar_norm = range_profile_sar / np.max(range_profile_sar)
        
        # Interpolate to common range axis for comparison
        common_range = np.linspace(800, 1200, 1000)  # Focus on target area
        lfmcw_interp = np.interp(common_range, range_axis, range_profile_lfmcw_norm)
        sar_interp = np.interp(common_range, time_axis, range_profile_sar_norm)
        
        ax5.plot(common_range, lfmcw_interp, 'b-', linewidth=2, label='LFMCW RDM', alpha=0.8)
        ax5.plot(common_range, sar_interp, 'g-', linewidth=2, label='Pulsed SAR', alpha=0.8)
        
        for target in targets:
            ax5.axvline(target['range'], color='red', linestyle='--', alpha=0.8, linewidth=2)
        
        ax5.set_xlabel('Range (m)')
        ax5.set_ylabel('Normalized Magnitude')
        ax5.set_title('Range Profile Comparison\n(Normalized)', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        
        # Plot 6: Processing Method Comparison Table
        ax6 = plt.subplot(2, 4, 6)
        
        comparison_data = [
            ['Parameter', 'LFMCW RDM', 'Pulsed SAR'],
            ['Signal Type', 'Continuous Wave', 'Pulsed'],
            ['Processing', '2D FFT (Range-Doppler)', 'Matched Filter + Azimuth'],
            ['Range Compression', 'Frequency Domain', 'Time/Freq Domain'],
            ['Doppler Processing', 'Direct FFT', 'Platform Motion'],
            ['Advantages', 'Simple, Real-time', 'Mature, Flexible'],
            ['Disadvantages', 'Limited Range', 'Complex Processing'],
            ['Range Resolution', f'{self.range_resolution:.3f} m', f'{self.range_resolution:.3f} m'],
            ['Max Range', f'{self.max_range/1e3:.1f} km', 'PRF Limited'],
            ['Processing Gain', 'Time-Bandwidth', 'Pulse Compression']
        ]
        
        table = ax6.table(cellText=comparison_data[1:], colLabels=comparison_data[0],
                         cellLoc='left', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style table
        for i in range(len(comparison_data)):
            for j in range(3):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white')
                elif j == 0:  # Parameter names
                    cell.set_facecolor('#E3F2FD')
                    cell.set_text_props(weight='bold')
                else:
                    cell.set_facecolor('#f8f9fa' if i % 2 == 0 else 'white')
        
        ax6.set_title('LFMCW vs Pulsed SAR\nComparison', fontweight='bold')
        ax6.axis('off')
        
        # Plot 7: Signal Processing Flow Diagrams
        ax7 = plt.subplot(2, 4, 7)
        
        # Create flow diagram text
        lfmcw_flow = [
            "LFMCW Processing Flow:",
            "1. Continuous LFM Transmission",
            "2. Receive Delayed Echoes", 
            "3. Range Compression (FFT)",
            "4. Doppler Processing (FFT)",
            "5. Range-Doppler Map (RDM)",
            "",
            "Key Features:",
            "• 2D FFT Processing",
            "• Direct Doppler Measurement",
            "• Real-time Capability",
            "• Simple Architecture"
        ]
        
        pulsed_flow = [
            "Pulsed SAR Processing Flow:",
            "1. Pulse Transmission",
            "2. Receive Echoes",
            "3. Pulse Compression",
            "4. Motion Compensation", 
            "5. Azimuth Compression",
            "6. Image Formation",
            "",
            "Key Features:",
            "• Matched Filtering",
            "• Synthetic Aperture",
            "• Motion Compensation",
            "• Complex Processing"
        ]
        
        # Display flows
        y_pos = 0.95
        for line in lfmcw_flow:
            if line.startswith("Key Features:") or line.startswith("LFMCW"):
                ax7.text(0.05, y_pos, line, transform=ax7.transAxes, fontweight='bold', fontsize=10)
            else:
                ax7.text(0.05, y_pos, line, transform=ax7.transAxes, fontsize=9)
            y_pos -= 0.07
        
        ax7.set_title('Processing Method Flows', fontweight='bold')
        ax7.axis('off')
        
        # Plot 8: Performance Metrics
        ax8 = plt.subplot(2, 4, 8)
        
        # Calculate some performance metrics
        lfmcw_snr = 10 * np.log10(np.max(rdm_magnitude) / np.mean(rdm_magnitude))
        sar_snr = 10 * np.log10(np.max(sar_magnitude) / np.mean(sar_magnitude))
        
        metrics_data = [
            ['Metric', 'LFMCW', 'Pulsed SAR', 'Winner'],
            ['SNR (dB)', f'{lfmcw_snr:.1f}', f'{sar_snr:.1f}', 'Pulsed SAR' if sar_snr > lfmcw_snr else 'LFMCW'],
            ['Processing Complexity', 'Low', 'High', 'LFMCW'],
            ['Real-time Capability', 'Excellent', 'Good', 'LFMCW'],
            ['Range Accuracy', 'Good', 'Excellent', 'Pulsed SAR'],
            ['Doppler Accuracy', 'Excellent', 'Good', 'LFMCW'],
            ['Power Efficiency', 'High', 'Medium', 'LFMCW'],
            ['Maturity', 'Medium', 'High', 'Pulsed SAR']
        ]
        
        table2 = ax8.table(cellText=metrics_data[1:], colLabels=metrics_data[0],
                          cellLoc='center', loc='center')
        table2.auto_set_font_size(False)
        table2.set_fontsize(9)
        table2.scale(1, 1.8)
        
        # Color code winners
        for i in range(1, len(metrics_data)):
            winner_col = table2[(i, 3)]
            if 'LFMCW' in metrics_data[i][3]:
                winner_col.set_facecolor('#E8F5E8')
            else:
                winner_col.set_facecolor('#FFF3E0')
        
        ax8.set_title('Performance Comparison', fontweight='bold')
        ax8.axis('off')
        
        plt.suptitle('LFMCW Range-Doppler Map vs Pulsed SAR Pulse Compression\nDetailed Analysis', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('../output/lfmcw_vs_pulsed_sar_analysis.png', dpi=300, bbox_inches='tight')
        print("\nLFMCW vs Pulsed SAR analysis plots saved to: ../output/lfmcw_vs_pulsed_sar_analysis.png")
        plt.close()
    
    def run_comparative_analysis(self):
        """Run comprehensive LFMCW vs Pulsed SAR analysis"""
        print("LFMCW vs PULSED SAR COMPARATIVE ANALYSIS")
        print("="*80)
        
        # Create output directory
        os.makedirs('../output', exist_ok=True)
        
        # Define test targets
        targets = [
            {'range': 1000.0, 'azimuth': -10.0, 'rcs': 1.0, 'name': 'Target_1'},
            {'range': 1050.0, 'azimuth': 10.0, 'rcs': 0.8, 'name': 'Target_2'}
        ]
        
        print(f"\nTest Scenario:")
        for target in targets:
            print(f"  {target['name']}: Range={target['range']:.1f}m, Azimuth={target['azimuth']:.1f}m, RCS={target['rcs']}")
        
        # Generate and process LFMCW signal
        print(f"\n--- LFMCW PROCESSING ---")
        lfmcw_signal, ref_chirp, t_sweep = self.generate_lfmcw_signal(targets)
        lfmcw_rdm, lfmcw_range_compressed = self.process_lfmcw_rdm(lfmcw_signal, ref_chirp)
        
        # Generate and process Pulsed SAR signal  
        print(f"\n--- PULSED SAR PROCESSING ---")
        pulse_responses, platform_positions, sar = self.generate_pulsed_sar_signal(targets)
        pulsed_sar_processed, pulsed_sar_raw = self.process_pulsed_sar(pulse_responses, sar)
        
        # Create comparison plots
        self.create_comparison_plots(lfmcw_rdm, lfmcw_range_compressed, 
                                   pulsed_sar_processed, pulsed_sar_raw, targets)
        
        # Analysis summary
        print(f"\n" + "="*80)
        print("COMPARATIVE ANALYSIS SUMMARY")
        print("="*80)
        
        print(f"\n1. SIGNAL CHARACTERISTICS:")
        print(f"   LFMCW:")
        print(f"   - Continuous transmission with frequency sweeps")
        print(f"   - Sweep time: {self.T_sweep*1e3:.1f} ms")
        print(f"   - Max unambiguous range: {self.max_range/1e3:.1f} km")
        print(f"   - Direct Doppler measurement")
        
        print(f"\n   Pulsed SAR:")
        print(f"   - Discrete pulse transmission")
        print(f"   - PRF: {self.PRF} Hz")
        print(f"   - Synthetic aperture formation")
        print(f"   - Motion compensation required")
        
        print(f"\n2. PROCESSING DIFFERENCES:")
        print(f"   LFMCW RDM:")
        print(f"   - Range compression: FFT correlation with reference")
        print(f"   - Doppler processing: FFT across sweeps")
        print(f"   - Output: Range-Doppler Map (RDM)")
        print(f"   - Processing: 2D FFT (efficient)")
        
        print(f"\n   Pulsed SAR:")
        print(f"   - Range compression: Matched filtering")
        print(f"   - Azimuth compression: Synthetic aperture processing")
        print(f"   - Output: SAR image")
        print(f"   - Processing: Complex multi-stage")
        
        print(f"\n3. ADVANTAGES & DISADVANTAGES:")
        print(f"   LFMCW Advantages:")
        print(f"   - Simple, real-time processing")
        print(f"   - Direct Doppler measurement")
        print(f"   - Continuous power transmission")
        print(f"   - Lower peak power requirements")
        
        print(f"   LFMCW Disadvantages:")
        print(f"   - Limited maximum range")
        print(f"   - Transmit/receive isolation challenges")
        print(f"   - Less mature for SAR applications")
        
        print(f"   Pulsed SAR Advantages:")
        print(f"   - Mature, well-established technology")
        print(f"   - Excellent range accuracy")
        print(f"   - Flexible system design")
        print(f"   - Better transmit/receive isolation")
        
        print(f"   Pulsed SAR Disadvantages:")
        print(f"   - Complex processing algorithms")
        print(f"   - Motion compensation required")
        print(f"   - Higher peak power requirements")
        print(f"   - Range-Doppler ambiguities")
        
        print(f"\n4. APPLICATION SUITABILITY:")
        print(f"   LFMCW RDM: Best for real-time, short-range, Doppler-sensitive applications")
        print(f"   Pulsed SAR: Best for high-resolution imaging, long-range surveillance")
        
        # Save detailed results
        analysis_results = {
            'timestamp': datetime.now().isoformat(),
            'analysis_type': 'lfmcw_vs_pulsed_sar',
            'system_parameters': {
                'carrier_frequency': self.fc,
                'bandwidth': self.B,
                'lfmcw_sweep_time': self.T_sweep,
                'pulsed_prf': self.PRF,
                'range_resolution': self.range_resolution,
                'max_range_lfmcw': self.max_range
            },
            'key_differences': {
                'signal_type': {
                    'lfmcw': 'Continuous wave with frequency modulation',
                    'pulsed': 'Discrete pulses with pulse compression'
                },
                'processing': {
                    'lfmcw': '2D FFT for Range-Doppler Map',
                    'pulsed': 'Matched filtering + synthetic aperture'
                },
                'output': {
                    'lfmcw': 'Range-Doppler Map (RDM)',
                    'pulsed': 'SAR image with azimuth compression'
                }
            },
            'performance_comparison': {
                'processing_complexity': {'winner': 'LFMCW', 'reason': 'Simple 2D FFT vs complex multi-stage'},
                'range_accuracy': {'winner': 'Pulsed SAR', 'reason': 'Mature pulse compression techniques'},
                'doppler_accuracy': {'winner': 'LFMCW', 'reason': 'Direct Doppler measurement'},
                'real_time_capability': {'winner': 'LFMCW', 'reason': 'Simpler processing pipeline'}
            }
        }
        
        with open('../output/lfmcw_vs_pulsed_sar_results.json', 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        print(f"\nDetailed analysis results saved to: ../output/lfmcw_vs_pulsed_sar_results.json")
        
        return lfmcw_rdm, pulsed_sar_processed, analysis_results

def main():
    """Main execution"""
    analysis = LFMCWvsPulsedSARAnalysis()
    lfmcw_rdm, pulsed_sar, results = analysis.run_comparative_analysis()
    return analysis, lfmcw_rdm, pulsed_sar, results

if __name__ == "__main__":
    analysis, lfmcw_rdm, pulsed_sar, results = main()
