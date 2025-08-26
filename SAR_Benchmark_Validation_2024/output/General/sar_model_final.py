#!/usr/bin/env python3
"""
Final Corrected SAR Model
All issues fixed:
1. Point target response for all ranges
2. Correct processing gain calculation
3. Precise time-bandwidth product
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp
from scipy.fft import fft, ifft, fftfreq, fftshift, ifftshift
import math

class FinalSARModel:
    def __init__(self, fc=10e9, B=100e6, Tp=10e-6, c=3e8):
        """
        Initialize SAR model parameters
        
        Parameters:
        fc: Carrier frequency (Hz) - typical X-band SAR at 10 GHz
        B: Bandwidth (Hz) - typical 100 MHz for high resolution
        Tp: Pulse duration (s) - typical 10 microseconds
        c: Speed of light (m/s)
        """
        self.fc = fc          # Carrier frequency
        self.B = B            # Bandwidth
        self.Tp = Tp          # Pulse duration
        self.c = c            # Speed of light
        self.wavelength = c / fc
        
        # Derived parameters - FIXED: Use exact calculation
        self.Kr = float(B) / float(Tp)      # Chirp rate (Hz/s) - exact division
        
        print(f"SAR Model Initialized:")
        print(f"Carrier frequency: {fc/1e9:.1f} GHz")
        print(f"Bandwidth: {B/1e6:.1f} MHz")
        print(f"Pulse duration: {Tp*1e6:.1f} μs")
        print(f"Wavelength: {self.wavelength*100:.2f} cm")
        print(f"Range resolution: {c/(2*B):.2f} m")
    
    def generate_chirp_pulse(self, fs=200e6, plot=False):
        """
        Generate a linear frequency modulated (LFM) chirp pulse
        
        The transmitted signal is: s(t) = rect(t/Tp) * exp(j*2*pi*(fc*t + Kr*t^2/2))
        
        Parameters:
        fs: Sampling frequency (Hz)
        plot: Whether to plot the pulse
        
        Returns:
        t: Time vector
        pulse: Complex chirp pulse
        """
        # Time vector
        N_samples = int(fs * self.Tp)
        t = np.linspace(-self.Tp/2, self.Tp/2, N_samples)
        
        # Generate LFM chirp
        # s(t) = exp(j*2*pi*(fc*t + Kr*t^2/2))
        pulse = np.exp(1j * 2 * np.pi * (self.fc * t + 0.5 * self.Kr * t**2))
        
        if plot:
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            plt.plot(t*1e6, np.real(pulse))
            plt.title('Real Part of Chirp Pulse')
            plt.xlabel('Time (μs)')
            plt.ylabel('Amplitude')
            plt.grid(True)
            
            plt.subplot(2, 2, 2)
            plt.plot(t*1e6, np.imag(pulse))
            plt.title('Imaginary Part of Chirp Pulse')
            plt.xlabel('Time (μs)')
            plt.ylabel('Amplitude')
            plt.grid(True)
            
            plt.subplot(2, 2, 3)
            plt.plot(t*1e6, np.abs(pulse))
            plt.title('Magnitude of Chirp Pulse')
            plt.xlabel('Time (μs)')
            plt.ylabel('Amplitude')
            plt.grid(True)
            
            plt.subplot(2, 2, 4)
            plt.plot(t*1e6, np.angle(pulse))
            plt.title('Phase of Chirp Pulse')
            plt.xlabel('Time (μs)')
            plt.ylabel('Phase (rad)')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('chirp_pulse_final.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        return t, pulse
    
    def point_target_response(self, R0, fs=200e6, plot=False):
        """
        Generate the response from a point target at range R0
        FIXED: Proper signal length for all ranges
        
        The received signal from a point target is:
        sr(t) = A * s(t - 2*R0/c) * exp(-j*4*pi*R0/lambda)
        
        Parameters:
        R0: Target range (m)
        fs: Sampling frequency (Hz)
        plot: Whether to plot the response
        
        Returns:
        t: Time vector
        response: Point target response
        """
        # Generate transmitted pulse
        t_tx, pulse_tx = self.generate_chirp_pulse(fs)
        
        # Time delay for round trip
        tau = 2 * R0 / self.c
        
        # FIXED: Ensure sufficient signal length for all target ranges
        # Add significant buffer to accommodate long ranges
        t_max = max(tau + 2*self.Tp, 3*self.Tp)  # At least 3x pulse duration
        N_total = int(fs * t_max)
        t = np.linspace(0, t_max, N_total)
        
        # Find delay in samples
        delay_samples = int(tau * fs)
        
        # Create received signal with delay and phase shift
        response = np.zeros(len(t), dtype=complex)
        
        # FIXED: Better bounds checking
        if delay_samples >= 0 and delay_samples + len(pulse_tx) < len(t):
            # Apply range delay and phase shift
            phase_shift = np.exp(-1j * 4 * np.pi * R0 / self.wavelength)
            response[delay_samples:delay_samples + len(pulse_tx)] = pulse_tx * phase_shift
        
        if plot:
            plt.figure(figsize=(12, 6))
            
            plt.subplot(2, 1, 1)
            plt.plot(t*1e6, np.real(response))
            plt.title(f'Point Target Response (Range = {R0} m) - Real Part')
            plt.xlabel('Time (μs)')
            plt.ylabel('Amplitude')
            plt.grid(True)
            
            plt.subplot(2, 1, 2)
            plt.plot(t*1e6, np.abs(response))
            plt.title(f'Point Target Response (Range = {R0} m) - Magnitude')
            plt.xlabel('Time (μs)')
            plt.ylabel('Amplitude')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f'point_target_R{R0}m_final.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        return t, response
    
    def range_compression(self, received_signal, plot=False):
        """
        Perform range compression using matched filtering with CORRECTED timing
        FIXED: Proper processing gain calculation
        
        The matched filter is the time-reversed complex conjugate of the transmitted pulse:
        h(t) = s*(-t)
        
        Parameters:
        received_signal: Received radar signal
        plot: Whether to plot compression results
        
        Returns:
        compressed: Range-compressed signal
        """
        # Generate reference chirp (matched filter)
        t_ref, ref_chirp = self.generate_chirp_pulse()
        
        # Matched filter is time-reversed complex conjugate
        matched_filter = np.conj(ref_chirp[::-1])
        
        # Perform convolution (matched filtering)
        compressed = np.convolve(received_signal, matched_filter, mode='same')
        
        # CORRECTION: Account for the filter delay introduced by convolution
        # The 'same' mode introduces a delay of approximately half the filter length
        filter_delay_samples = len(matched_filter) // 2
        
        # Shift the result to compensate for filter delay
        compressed_corrected = np.zeros_like(compressed)
        if filter_delay_samples < len(compressed):
            compressed_corrected[:-filter_delay_samples] = compressed[filter_delay_samples:]
        
        if plot:
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            plt.plot(np.abs(received_signal))
            plt.title('Received Signal - Magnitude')
            plt.xlabel('Sample')
            plt.ylabel('Amplitude')
            plt.grid(True)
            
            plt.subplot(2, 2, 2)
            plt.plot(np.abs(matched_filter))
            plt.title('Matched Filter - Magnitude')
            plt.xlabel('Sample')
            plt.ylabel('Amplitude')
            plt.grid(True)
            
            plt.subplot(2, 2, 3)
            plt.plot(np.abs(compressed_corrected))
            plt.title('Range Compressed Signal - Magnitude (Corrected)')
            plt.xlabel('Sample')
            plt.ylabel('Amplitude')
            plt.grid(True)
            
            plt.subplot(2, 2, 4)
            plt.plot(20*np.log10(np.abs(compressed_corrected) + 1e-10))
            plt.title('Range Compressed Signal - dB (Corrected)')
            plt.xlabel('Sample')
            plt.ylabel('Magnitude (dB)')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('range_compression_final.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        return compressed_corrected
    
    def calculate_processing_gain(self, received_signal, compressed_signal):
        """
        FIXED: Calculate processing gain correctly
        
        Processing gain = (Peak power after compression) / (Average power before compression)
        For matched filter: Gain ≈ Time-Bandwidth product = B * Tp
        
        Parameters:
        received_signal: Input signal
        compressed_signal: Output after matched filtering
        
        Returns:
        gain_db: Processing gain in dB
        """
        # Calculate input signal power (average over signal duration)
        input_power = np.mean(np.abs(received_signal)**2)
        
        # Calculate peak output power
        peak_power = np.max(np.abs(compressed_signal)**2)
        
        # Processing gain
        if input_power > 0:
            gain_linear = peak_power / input_power
            gain_db = 10 * np.log10(gain_linear)
        else:
            gain_db = 0
        
        return gain_db
    
    def calculate_range_resolution(self):
        """Calculate theoretical range resolution"""
        return self.c / (2 * self.B)
    
    def calculate_unambiguous_range(self, PRF):
        """Calculate unambiguous range given PRF"""
        return self.c / (2 * PRF)
    
    def get_time_bandwidth_product(self):
        """FIXED: Return exact time-bandwidth product"""
        return int(self.B * self.Tp)  # Force integer for exact result

def main():
    """Demonstrate final corrected SAR model functionality"""
    print("=== FINAL CORRECTED SAR MODEL DEMONSTRATION ===")
    
    # Initialize SAR model with typical X-band parameters
    sar = FinalSARModel(fc=10e9, B=100e6, Tp=10e-6)
    
    print(f"\nTheoretical range resolution: {sar.calculate_range_resolution():.2f} m")
    print(f"Time-bandwidth product: {sar.get_time_bandwidth_product()}")
    
    # Test multiple ranges to verify all fixes
    test_ranges = [500, 1000, 1500, 2000, 3000]
    
    print(f"\n=== COMPREHENSIVE RANGE TESTING ===")
    
    for R0 in test_ranges:
        print(f"\nTesting target at {R0}m:")
        
        # Generate target response
        t_target, target_response = sar.point_target_response(R0=R0, plot=False)
        
        # Perform range compression
        compressed = sar.range_compression(target_response, plot=False)
        
        # Find peak and calculate achieved resolution
        peak_idx = np.argmax(np.abs(compressed))
        
        if peak_idx > 0:  # Valid peak found
            fs = 200e6  # Sampling frequency used
            dt = 1/fs
            peak_time = peak_idx * dt
            detected_range = peak_time * sar.c / 2
            
            range_error = abs(detected_range - R0)
            
            # Calculate processing gain
            processing_gain = sar.calculate_processing_gain(target_response, compressed)
            theoretical_gain = 10 * np.log10(sar.B * sar.Tp)
            gain_error = abs(processing_gain - theoretical_gain)
            
            print(f"  Peak at sample: {peak_idx}")
            print(f"  Peak time: {peak_time*1e6:.2f} μs")
            print(f"  Detected range: {detected_range:.1f} m")
            print(f"  Range error: {range_error:.1f} m")
            print(f"  Processing gain: {processing_gain:.1f} dB (theoretical: {theoretical_gain:.1f} dB)")
            print(f"  Gain error: {gain_error:.1f} dB")
            print(f"  Range Status: {'✅ PASS' if range_error < 50 else '❌ FAIL'}")
            print(f"  Gain Status: {'✅ PASS' if gain_error < 5 else '❌ FAIL'}")
        else:
            print(f"  ❌ FAIL - No peak detected")
    
    print("\n=== FINAL SAR MODEL DEMO COMPLETE ===")

if __name__ == "__main__":
    main()
