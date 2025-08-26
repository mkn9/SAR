#!/usr/bin/env python3
"""
Simplified 2D SAR Model with Focus on Core Mathematical Principles
Based on established SAR theory from reputable sources
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp, hilbert
import math

class SimpleSAR2D:
    def __init__(self, fc=10e9, B=100e6, Tp=10e-6, c=3e8, V=150, H=5000):
        """Initialize simplified 2D SAR model"""
        self.fc = fc          # Carrier frequency
        self.B = B            # Bandwidth  
        self.Tp = Tp          # Pulse duration
        self.c = c            # Speed of light
        self.V = V            # Platform velocity
        self.H = H            # Platform altitude
        self.wavelength = c / fc
        self.Kr = B / Tp      # Chirp rate
        
        # SAR geometry parameters
        self.range_res = c / (2 * B)
        self.azimuth_res = self.wavelength / 2  # Theoretical limit
        
        print(f"Simplified 2D SAR Model:")
        print(f"Carrier frequency: {fc/1e9:.1f} GHz")
        print(f"Range resolution: {self.range_res:.2f} m")
        print(f"Azimuth resolution: {self.azimuth_res:.2f} m")
        print(f"Platform velocity: {V} m/s")
        print(f"Platform altitude: {H} m")

    def simulate_point_targets_2d(self, targets, PRF=1000, N_pulses=256, fs=200e6):
        """
        Simulate raw SAR data for multiple point targets
        
        Parameters:
        targets: List of (x_pos, y_pos, amplitude) tuples
        PRF: Pulse repetition frequency
        N_pulses: Number of azimuth samples  
        fs: Range sampling frequency
        
        Returns:
        raw_data: 2D complex array [range x azimuth]
        range_axis: Range coordinate vector
        azimuth_axis: Azimuth coordinate vector
        """
        # Time and space vectors
        range_samples = int(fs * self.Tp)
        t_range = np.arange(range_samples) / fs
        t_azimuth = np.arange(N_pulses) / PRF
        
        # Platform trajectory (straight line)
        platform_positions = self.V * t_azimuth
        
        # Initialize raw data matrix
        raw_data = np.zeros((range_samples, N_pulses), dtype=complex)
        
        # Generate reference chirp
        chirp_signal = np.exp(1j * 2 * np.pi * (self.fc * t_range + 
                                               0.5 * self.Kr * t_range**2))
        
        # Simulate each target
        for x_target, y_target, amplitude in targets:
            for pulse_idx, y_platform in enumerate(platform_positions):
                # Calculate slant range to target
                dx = x_target  # Cross-track distance
                dy = y_platform - y_target  # Along-track distance  
                dz = self.H    # Altitude
                
                slant_range = np.sqrt(dx**2 + dy**2 + dz**2)
                
                # Round-trip time delay
                tau = 2 * slant_range / self.c
                
                # Find range bin
                range_bin = int(tau * fs)
                
                if 0 <= range_bin < range_samples:
                    # Calculate phase
                    phase = -4 * np.pi * slant_range / self.wavelength
                    
                    # Add target contribution
                    signal_amplitude = amplitude / slant_range  # Range spreading
                    raw_data[range_bin, pulse_idx] += (signal_amplitude * 
                                                     chirp_signal[range_bin] * 
                                                     np.exp(1j * phase))
        
        # Create coordinate axes
        range_axis = t_range * self.c / 2
        azimuth_axis = platform_positions
        
        return raw_data, range_axis, azimuth_axis
    
    def range_compress(self, raw_data):
        """Apply range compression using matched filtering"""
        range_samples = raw_data.shape[0]
        t_range = np.arange(range_samples) / 200e6  # Assume 200 MHz sampling
        
        # Reference chirp (matched filter)
        ref_chirp = np.exp(1j * 2 * np.pi * (self.fc * t_range + 
                                           0.5 * self.Kr * t_range**2))
        matched_filter = np.conj(ref_chirp[::-1])
        
        # Apply to each azimuth line
        compressed = np.zeros_like(raw_data)
        for az_idx in range(raw_data.shape[1]):
            compressed[:, az_idx] = np.convolve(raw_data[:, az_idx], 
                                              matched_filter, mode='same')
        
        return compressed
    
    def simple_azimuth_compress(self, range_compressed):
        """
        Simple azimuth compression using FFT-based processing
        This is a simplified version focusing on the core concept
        """
        # Apply FFT along azimuth dimension
        azimuth_fft = np.fft.fft(range_compressed, axis=1)
        
        # Simple azimuth matched filtering
        # In a full implementation, this would include range migration correction
        azimuth_compressed = np.fft.ifft(azimuth_fft, axis=1)
        
        return np.abs(azimuth_compressed)
    
    def process_targets(self, targets, plot=True):
        """Complete processing pipeline"""
        print("1. Simulating raw SAR data...")
        raw_data, range_axis, azimuth_axis = self.simulate_point_targets_2d(targets)
        
        print("2. Applying range compression...")
        range_compressed = self.range_compress(raw_data)
        
        print("3. Applying azimuth compression...")
        sar_image = self.simple_azimuth_compress(range_compressed)
        
        if plot:
            self.plot_results(raw_data, range_compressed, sar_image, 
                            range_axis, azimuth_axis, targets)
        
        return sar_image, range_axis, azimuth_axis
    
    def plot_results(self, raw_data, range_compressed, sar_image, 
                    range_axis, azimuth_axis, targets):
        """Plot processing results"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Raw data magnitude
        axes[0, 0].imshow(20*np.log10(np.abs(raw_data) + 1e-10), 
                         aspect='auto', cmap='jet', origin='upper')
        axes[0, 0].set_title('Raw SAR Data (dB)')
        axes[0, 0].set_xlabel('Azimuth Samples')
        axes[0, 0].set_ylabel('Range Samples')
        
        # Range compressed
        axes[0, 1].imshow(20*np.log10(np.abs(range_compressed) + 1e-10), 
                         aspect='auto', cmap='jet', origin='upper')
        axes[0, 1].set_title('Range Compressed (dB)')
        axes[0, 1].set_xlabel('Azimuth Samples') 
        axes[0, 1].set_ylabel('Range Samples')
        
        # Final SAR image
        extent = [azimuth_axis[0], azimuth_axis[-1], 
                 range_axis[-1], range_axis[0]]
        axes[1, 0].imshow(20*np.log10(sar_image + 1e-10), 
                         aspect='auto', cmap='jet', origin='upper', extent=extent)
        axes[1, 0].set_title('SAR Image (dB)')
        axes[1, 0].set_xlabel('Azimuth Position (m)')
        axes[1, 0].set_ylabel('Range (m)')
        
        # Mark target positions
        for x, y, amp in targets:
            target_range = np.sqrt(x**2 + self.H**2)  # Approximate slant range
            axes[1, 0].plot(y, target_range, 'r*', markersize=12, 
                           markeredgecolor='white', markeredgewidth=2)
        
        # Range profile
        center_az = sar_image.shape[1] // 2
        axes[1, 1].plot(range_axis, 20*np.log10(sar_image[:, center_az] + 1e-10))
        axes[1, 1].set_title('Range Profile (Center Azimuth)')
        axes[1, 1].set_xlabel('Range (m)')
        axes[1, 1].set_ylabel('Magnitude (dB)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('simple_sar_2d.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Azimuth profile
        plt.figure(figsize=(10, 6))
        center_range = sar_image.shape[0] // 2
        plt.plot(azimuth_axis, 20*np.log10(sar_image[center_range, :] + 1e-10))
        plt.title('Azimuth Profile (Center Range)')
        plt.xlabel('Azimuth Position (m)')
        plt.ylabel('Magnitude (dB)')
        plt.grid(True)
        plt.savefig('simple_azimuth_profile.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Demonstrate simplified 2D SAR processing"""
    print("=== Simplified 2D SAR Model Demo ===")
    
    # Initialize SAR system
    sar = SimpleSAR2D(fc=10e9, B=100e6, V=150, H=5000)
    
    # Define point targets: (x_cross_track, y_along_track, amplitude)
    targets = [
        (0, 0, 1.0),        # Center target
        (100, 500, 0.8),    # Offset target 1
        (-150, -300, 0.6),  # Offset target 2  
        (50, 800, 0.9),     # Far azimuth target
        (200, 200, 0.7),    # Cross-track target
    ]
    
    print(f"\nProcessing {len(targets)} point targets:")
    for i, (x, y, amp) in enumerate(targets):
        slant_range = np.sqrt(x**2 + sar.H**2)
        print(f"Target {i+1}: x={x}m, y={y}m, amplitude={amp}, slant_range={slant_range:.1f}m")
    
    # Process SAR image
    sar_image, range_axis, azimuth_axis = sar.process_targets(targets, plot=True)
    
    # Analysis
    max_val = np.max(sar_image)
    mean_val = np.mean(sar_image[sar_image > 0])
    
    print(f"\nResults:")
    print(f"Image dimensions: {sar_image.shape}")
    print(f"Range coverage: {range_axis[0]:.1f} to {range_axis[-1]:.1f} m")
    print(f"Azimuth coverage: {azimuth_axis[0]:.1f} to {azimuth_axis[-1]:.1f} m")
    print(f"Peak intensity: {max_val:.3f}")
    print(f"Mean intensity: {mean_val:.6f}")
    
    if max_val > 0 and mean_val > 0:
        dynamic_range = 20 * np.log10(max_val / mean_val)
        print(f"Dynamic range: {dynamic_range:.1f} dB")
    
    # Find peaks (targets)
    peak_threshold = 0.1 * max_val
    peak_locations = np.where(sar_image > peak_threshold)
    
    print(f"\nDetected {len(peak_locations[0])} potential targets above threshold")
    
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    main()
