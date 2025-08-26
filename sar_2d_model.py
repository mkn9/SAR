#!/usr/bin/env python3
"""
Enhanced 2D Synthetic Aperture Radar (SAR) Model
Includes both range and azimuth compression for full SAR image formation

Based on mathematical formulas from reputable sources:
- Introduction to Synthetic Aperture Radar by Ian G. Cumming and Frank H. Wong
- Synthetic Aperture Radar Signal Processing by Mehrdad Soumekh
- Digital Processing of Synthetic Aperture Radar Data by Ian G. Cumming
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp
from scipy.fft import fft, ifft, fftfreq, fftshift, ifftshift
import math

class SAR2DModel:
    def __init__(self, fc=10e9, B=100e6, Tp=10e-6, c=3e8, V=150, H=5000):
        """
        Initialize 2D SAR model parameters
        
        Parameters:
        fc: Carrier frequency (Hz) - X-band at 10 GHz
        B: Bandwidth (Hz) - 100 MHz for high resolution
        Tp: Pulse duration (s) - 10 microseconds
        c: Speed of light (m/s)
        V: Platform velocity (m/s) - typical 150 m/s
        H: Platform altitude (m) - typical 5000 m
        """
        self.fc = fc          # Carrier frequency
        self.B = B            # Bandwidth
        self.Tp = Tp          # Pulse duration
        self.c = c            # Speed of light
        self.V = V            # Platform velocity
        self.H = H            # Platform altitude
        self.wavelength = c / fc
        
        # Derived parameters
        self.Kr = B / Tp      # Range chirp rate (Hz/s)
        
        # Azimuth parameters
        self.La = self.wavelength * H / (2 * V)  # Synthetic aperture length
        self.Ta = self.La / V                    # Synthetic aperture time
        
        print(f"2D SAR Model Initialized:")
        print(f"Carrier frequency: {fc/1e9:.1f} GHz")
        print(f"Bandwidth: {B/1e6:.1f} MHz")
        print(f"Platform velocity: {V} m/s")
        print(f"Platform altitude: {H} m")
        print(f"Wavelength: {self.wavelength*100:.2f} cm")
        print(f"Range resolution: {c/(2*B):.2f} m")
        print(f"Azimuth resolution: {self.wavelength*H/(2*self.La):.2f} m")
        print(f"Synthetic aperture length: {self.La:.1f} m")
        print(f"Synthetic aperture time: {self.Ta:.2f} s")
    
    def generate_raw_data(self, targets, PRF=1000, fs=200e6, N_pulses=1024):
        """
        Generate raw SAR data for multiple point targets
        
        Parameters:
        targets: List of tuples (x, y, reflectivity) for each target
        PRF: Pulse repetition frequency (Hz)
        fs: Sampling frequency (Hz)
        N_pulses: Number of pulses in synthetic aperture
        
        Returns:
        raw_data: 2D array [range_samples x azimuth_samples]
        range_axis: Range axis in meters
        azimuth_axis: Azimuth axis in meters
        """
        # Time vectors
        range_time = np.arange(int(fs * self.Tp)) / fs
        azimuth_time = np.arange(N_pulses) / PRF
        
        # Initialize raw data matrix
        N_range = len(range_time)
        raw_data = np.zeros((N_range, N_pulses), dtype=complex)
        
        # Generate reference chirp
        ref_chirp = np.exp(1j * 2 * np.pi * (self.fc * range_time + 
                                           0.5 * self.Kr * range_time**2))
        
        # Platform positions
        platform_y = self.V * azimuth_time
        platform_x = 0  # Assume straight flight path
        platform_z = self.H
        
        for target_x, target_y, reflectivity in targets:
            for pulse_idx, t_az in enumerate(azimuth_time):
                # Calculate instantaneous range
                dx = platform_x - target_x
                dy = platform_y[pulse_idx] - target_y
                dz = platform_z
                R_inst = np.sqrt(dx**2 + dy**2 + dz**2)
                
                # Range delay
                tau = 2 * R_inst / self.c
                
                # Find corresponding range bin
                range_bin = int(tau * fs)
                
                if 0 <= range_bin < N_range:
                    # Phase due to range
                    phase = -4 * np.pi * R_inst / self.wavelength
                    
                    # Add target response to raw data
                    if range_bin < len(ref_chirp):
                        raw_data[range_bin, pulse_idx] += (reflectivity * 
                                                         ref_chirp[range_bin] * 
                                                         np.exp(1j * phase))
        
        # Create axes
        range_axis = range_time * self.c / 2  # Convert to range in meters
        azimuth_axis = platform_y
        
        return raw_data, range_axis, azimuth_axis
    
    def range_compression(self, raw_data):
        """
        Perform range compression using matched filtering
        
        Parameters:
        raw_data: Raw SAR data [range x azimuth]
        
        Returns:
        range_compressed: Range-compressed data
        """
        # Generate matched filter (time-reversed conjugate of chirp)
        range_time = np.arange(raw_data.shape[0]) / 200e6  # Assuming 200 MHz sampling
        ref_chirp = np.exp(1j * 2 * np.pi * (self.fc * range_time + 
                                           0.5 * self.Kr * range_time**2))
        matched_filter = np.conj(ref_chirp[::-1])
        
        # Apply matched filter to each azimuth line
        range_compressed = np.zeros_like(raw_data)
        for az_idx in range(raw_data.shape[1]):
            range_compressed[:, az_idx] = np.convolve(raw_data[:, az_idx], 
                                                    matched_filter, mode='same')
        
        return range_compressed
    
    def azimuth_compression(self, range_compressed_data, range_axis, azimuth_axis):
        """
        Perform azimuth compression using Range Doppler Algorithm (RDA)
        
        Parameters:
        range_compressed_data: Range-compressed data
        range_axis: Range axis in meters
        azimuth_axis: Azimuth axis in meters
        
        Returns:
        sar_image: Final SAR image
        """
        N_range, N_azimuth = range_compressed_data.shape
        
        # Transform to range-Doppler domain
        range_doppler = np.zeros_like(range_compressed_data)
        for range_idx in range(N_range):
            range_doppler[range_idx, :] = fft(range_compressed_data[range_idx, :])
        
        # Doppler frequency axis
        PRF = len(azimuth_axis) / (azimuth_axis[-1] - azimuth_axis[0]) * self.V
        doppler_freq = fftfreq(N_azimuth, 1/PRF)
        doppler_freq = fftshift(doppler_freq)
        range_doppler = fftshift(range_doppler, axes=1)
        
        # Range migration correction and azimuth compression
        sar_image = np.zeros_like(range_doppler)
        
        for range_idx, R0 in enumerate(range_axis):
            if R0 > 0:  # Avoid division by zero
                for doppler_idx, fd in enumerate(doppler_freq):
                    # Calculate range curvature
                    # R(t) ≈ R0 + (λ*fd*t)²/(8*R0) for small squint angles
                    range_curve = (self.wavelength * fd)**2 / (8 * R0)
                    
                    # Apply range migration correction
                    migration_samples = int(2 * range_curve / self.c * 200e6)  # Convert to samples
                    
                    if 0 <= range_idx + migration_samples < N_range:
                        sar_image[range_idx, doppler_idx] = range_doppler[range_idx + migration_samples, doppler_idx]
        
        # Transform back to time domain for final image
        final_image = np.zeros_like(sar_image)
        for range_idx in range(N_range):
            final_image[range_idx, :] = ifft(ifftshift(sar_image[range_idx, :]))
        
        return np.abs(final_image)
    
    def process_sar_image(self, targets, PRF=1000, fs=200e6, N_pulses=1024, plot=True):
        """
        Complete SAR processing pipeline
        
        Parameters:
        targets: List of target positions and reflectivities
        PRF: Pulse repetition frequency
        fs: Sampling frequency
        N_pulses: Number of pulses
        plot: Whether to generate plots
        
        Returns:
        sar_image: Final processed SAR image
        """
        print("1. Generating raw SAR data...")
        raw_data, range_axis, azimuth_axis = self.generate_raw_data(
            targets, PRF, fs, N_pulses)
        
        print("2. Performing range compression...")
        range_compressed = self.range_compression(raw_data)
        
        print("3. Performing azimuth compression...")
        sar_image = self.azimuth_compression(range_compressed, range_axis, azimuth_axis)
        
        if plot:
            self.plot_processing_results(raw_data, range_compressed, sar_image, 
                                       range_axis, azimuth_axis, targets)
        
        return sar_image, range_axis, azimuth_axis
    
    def plot_processing_results(self, raw_data, range_compressed, sar_image, 
                              range_axis, azimuth_axis, targets):
        """Plot SAR processing results"""
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Raw data
        axes[0, 0].imshow(20*np.log10(np.abs(raw_data) + 1e-10), 
                         aspect='auto', cmap='jet', origin='upper')
        axes[0, 0].set_title('Raw SAR Data (dB)')
        axes[0, 0].set_xlabel('Azimuth Samples')
        axes[0, 0].set_ylabel('Range Samples')
        
        # Range compressed
        axes[0, 1].imshow(20*np.log10(np.abs(range_compressed) + 1e-10), 
                         aspect='auto', cmap='jet', origin='upper')
        axes[0, 1].set_title('Range Compressed Data (dB)')
        axes[0, 1].set_xlabel('Azimuth Samples')
        axes[0, 1].set_ylabel('Range Samples')
        
        # Final SAR image
        extent = [azimuth_axis[0], azimuth_axis[-1], 
                 range_axis[-1], range_axis[0]]
        im = axes[1, 0].imshow(20*np.log10(sar_image + 1e-10), 
                              aspect='auto', cmap='jet', origin='upper',
                              extent=extent)
        axes[1, 0].set_title('Final SAR Image (dB)')
        axes[1, 0].set_xlabel('Azimuth (m)')
        axes[1, 0].set_ylabel('Range (m)')
        
        # Plot target positions
        for target_x, target_y, _ in targets:
            axes[1, 0].plot(target_y, np.sqrt(target_x**2 + self.H**2), 
                           'r*', markersize=10, markeredgecolor='white')
        
        # Range profile at center azimuth
        center_az = sar_image.shape[1] // 2
        axes[1, 1].plot(range_axis, 20*np.log10(sar_image[:, center_az] + 1e-10))
        axes[1, 1].set_title('Range Profile (Center Azimuth)')
        axes[1, 1].set_xlabel('Range (m)')
        axes[1, 1].set_ylabel('Magnitude (dB)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('sar_2d_processing.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Additional plot: Azimuth profile
        plt.figure(figsize=(10, 6))
        center_range = sar_image.shape[0] // 2
        plt.plot(azimuth_axis, 20*np.log10(sar_image[center_range, :] + 1e-10))
        plt.title('Azimuth Profile (Center Range)')
        plt.xlabel('Azimuth (m)')
        plt.ylabel('Magnitude (dB)')
        plt.grid(True)
        plt.savefig('azimuth_profile.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Demonstrate 2D SAR model functionality"""
    print("=== 2D SAR Model Demonstration ===")
    
    # Initialize 2D SAR model
    sar = SAR2DModel(fc=10e9, B=100e6, Tp=10e-6, V=150, H=5000)
    
    # Define point targets: (x, y, reflectivity)
    # x is cross-track, y is along-track, reflectivity is target strength
    targets = [
        (0, 0, 1.0),      # Target at scene center
        (100, 200, 0.8),  # Target offset in both dimensions
        (-50, -100, 0.6), # Target in opposite quadrant
        (0, 400, 0.9),    # Target along-track only
        (200, 0, 0.7),    # Target cross-track only
    ]
    
    print(f"\nProcessing {len(targets)} point targets:")
    for i, (x, y, refl) in enumerate(targets):
        print(f"Target {i+1}: x={x}m, y={y}m, reflectivity={refl}")
    
    # Process SAR image
    sar_image, range_axis, azimuth_axis = sar.process_sar_image(
        targets, PRF=1000, N_pulses=512, plot=True)
    
    # Calculate image statistics
    peak_value = np.max(sar_image)
    mean_value = np.mean(sar_image)
    std_value = np.std(sar_image)
    
    print(f"\nSAR Image Statistics:")
    print(f"Image size: {sar_image.shape[0]} x {sar_image.shape[1]}")
    print(f"Range extent: {range_axis[0]:.1f} to {range_axis[-1]:.1f} m")
    print(f"Azimuth extent: {azimuth_axis[0]:.1f} to {azimuth_axis[-1]:.1f} m")
    print(f"Peak value: {peak_value:.3f}")
    print(f"Mean value: {mean_value:.6f}")
    print(f"Std deviation: {std_value:.6f}")
    print(f"Dynamic range: {20*np.log10(peak_value/mean_value):.1f} dB")
    
    print("\n=== 2D SAR Model Demo Complete ===")

if __name__ == "__main__":
    main()
