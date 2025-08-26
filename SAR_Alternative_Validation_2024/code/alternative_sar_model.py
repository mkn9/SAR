#!/usr/bin/env python3
"""
Alternative SAR Model Implementation
Based on different mathematical formulations and sources including:
- Skolnik's Radar Handbook (3rd Edition) - Chapter 23: Synthetic Aperture Radar
- Richards' "Principles of Modern Radar: Basic Principles" - SAR Processing
- Omega-K Algorithm (Range Migration Algorithm) concepts
- Alternative signal processing approaches for cross-validation

Key Differences from Primary Model:
1. Different chirp generation approach (frequency domain emphasis)
2. Alternative range compression using frequency domain correlation
3. Omega-K inspired processing steps
4. Different parameter definitions and calculations
5. Alternative resolution formulas for validation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp, hilbert
from scipy.fft import fft, ifft, fftfreq, fftshift, ifftshift
import math

class AlternativeSARModel:
    def __init__(self, fc=10e9, B=100e6, Tp=10e-6, c=3e8, PRF=1000):
        """
        Initialize Alternative SAR model with different parameter emphasis
        
        Based on Skolnik Radar Handbook approach:
        - fc: Center frequency (Hz) 
        - B: Bandwidth (Hz)
        - Tp: Pulse duration (s)
        - c: Speed of light (m/s)
        - PRF: Pulse Repetition Frequency (Hz)
        
        Alternative formulations from Richards and Omega-K literature
        """
        # Primary parameters
        self.fc = fc
        self.B = B
        self.Tp = Tp
        self.c = c
        self.PRF = PRF
        
        # Derived parameters using alternative formulations
        self.wavelength = c / fc  # λ = c/fc (Skolnik)
        self.chirp_rate = B / Tp  # Kr = B/Tp (Richards)
        
        # Alternative resolution calculations (Skolnik formulation)
        self.range_resolution_skolnik = c / (2 * B)  # Skolnik Eq. 23.1
        
        # Time-bandwidth product (Richards emphasis)
        self.time_bandwidth_product = B * Tp
        
        # Sampling parameters for Omega-K processing
        self.fs = 2 * B  # Nyquist sampling for bandwidth
        
        # Print initialization
        print(f"Alternative SAR Model Initialized (Skolnik/Richards/Omega-K):")
        print(f"Center frequency: {fc/1e9:.1f} GHz")
        print(f"Bandwidth: {B/1e6:.1f} MHz") 
        print(f"Pulse duration: {Tp*1e6:.1f} μs")
        print(f"Wavelength: {self.wavelength*100:.2f} cm")
        print(f"Range resolution (Skolnik): {self.range_resolution_skolnik:.2f} m")
        print(f"Time-bandwidth product: {self.time_bandwidth_product:.0f}")
    
    def generate_frequency_domain_chirp(self, plot=False):
        """
        Generate LFM chirp using frequency domain approach
        Alternative to time-domain generation - emphasizes spectral properties
        
        Based on Richards' frequency domain SAR processing approach
        """
        # Time vector
        t = np.arange(-self.Tp/2, self.Tp/2, 1/self.fs)
        
        # Frequency domain chirp generation (Richards approach)
        # Emphasizes spectral shaping and phase characteristics
        f = fftfreq(len(t), 1/self.fs)
        
        # Frequency domain representation of chirp
        # H(f) = rect(f/B) * exp(-j*π*f²/Kr) (Richards formulation)
        H_freq = np.zeros(len(f), dtype=complex)
        freq_mask = np.abs(f) <= self.B/2
        H_freq[freq_mask] = np.exp(-1j * np.pi * f[freq_mask]**2 / self.chirp_rate)
        
        # Convert to time domain
        h_time = ifft(ifftshift(H_freq))
        
        # Apply window function (alternative to rectangular - Skolnik recommendation)
        window = np.exp(-2 * (t / self.Tp)**2)  # Gaussian window
        chirp_signal = h_time * window
        
        if plot:
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            plt.plot(t*1e6, np.real(chirp_signal), 'b-', linewidth=1)
            plt.title('Alternative Chirp - Real Part')
            plt.xlabel('Time (μs)')
            plt.ylabel('Amplitude')
            plt.grid(True)
            
            plt.subplot(2, 2, 2)
            plt.plot(t*1e6, np.imag(chirp_signal), 'r-', linewidth=1)
            plt.title('Alternative Chirp - Imaginary Part')
            plt.xlabel('Time (μs)')
            plt.ylabel('Amplitude')
            plt.grid(True)
            
            plt.subplot(2, 2, 3)
            freq_MHz = f/1e6
            plt.plot(freq_MHz, np.abs(H_freq), 'g-', linewidth=1)
            plt.title('Frequency Domain Representation')
            plt.xlabel('Frequency (MHz)')
            plt.ylabel('Magnitude')
            plt.grid(True)
            plt.xlim(-self.B/1e6, self.B/1e6)
            
            plt.subplot(2, 2, 4)
            plt.plot(t*1e6, np.abs(chirp_signal), 'm-', linewidth=1)
            plt.title('Chirp Envelope')
            plt.xlabel('Time (μs)')
            plt.ylabel('Magnitude')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('../output/alternative_chirp_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        return t, chirp_signal
    
    def point_target_response_omega_k(self, R0=1000, sigma=1.0, plot=False):
        """
        Generate point target response using Omega-K inspired approach
        Alternative formulation emphasizing 2D frequency domain processing
        
        Based on Omega-K algorithm concepts from SAR literature
        """
        # Generate transmitted chirp
        t_tx, chirp_tx = self.generate_frequency_domain_chirp()
        
        # Calculate round-trip delay
        tau = 2 * R0 / self.c
        
        # Create longer time vector for received signal
        t_total = 4 * self.Tp  # Extended time window
        t_rx = np.arange(0, t_total, 1/self.fs)
        
        # Initialize received signal
        received_signal = np.zeros(len(t_rx), dtype=complex)
        
        # Calculate delay in samples
        delay_samples = int(tau * self.fs)
        
        if delay_samples < len(t_rx) - len(chirp_tx):
            # Apply range delay and phase shift (Omega-K formulation)
            # Phase includes both range delay and two-way propagation
            phase_shift = np.exp(-1j * 4 * np.pi * R0 / self.wavelength)
            
            # Range migration effect (simplified Omega-K concept)
            range_migration_factor = np.exp(-1j * 2 * np.pi * self.fc * tau)
            
            # Combined phase correction
            total_phase = phase_shift * range_migration_factor
            
            # Insert delayed and phase-shifted chirp
            end_idx = delay_samples + len(chirp_tx)
            received_signal[delay_samples:end_idx] = sigma * chirp_tx * total_phase
        
        if plot:
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            plt.plot(t_rx*1e6, np.real(received_signal), 'b-', linewidth=1)
            plt.title(f'Point Target Response (R0={R0}m) - Real')
            plt.xlabel('Time (μs)')
            plt.ylabel('Amplitude')
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            plt.plot(t_rx*1e6, np.abs(received_signal), 'r-', linewidth=1)
            plt.title(f'Point Target Response - Magnitude')
            plt.xlabel('Time (μs)')
            plt.ylabel('Magnitude')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f'../output/alternative_point_target_R{R0}m.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        return t_rx, received_signal
    
    def frequency_domain_range_compression(self, received_signal, plot=False):
        """
        Range compression using frequency domain correlation
        Alternative to time-domain convolution - Omega-K inspired
        
        Based on frequency domain processing from Richards and Omega-K literature
        """
        # Generate reference chirp for matched filtering
        t_ref, ref_chirp = self.generate_frequency_domain_chirp()
        
        # Pad signals to same length for frequency domain processing
        N = len(received_signal)
        if len(ref_chirp) != N:
            # Zero-pad reference chirp to match received signal length
            ref_padded = np.zeros(N, dtype=complex)
            ref_padded[:len(ref_chirp)] = ref_chirp
            ref_chirp = ref_padded
        
        # Frequency domain correlation (alternative to time domain)
        # Matched filter: H*(f) where H(f) is chirp spectrum
        received_fft = fft(received_signal)
        ref_fft_conj = np.conj(fft(ref_chirp))
        
        # Apply frequency domain matched filter
        compressed_fft = received_fft * ref_fft_conj
        
        # Additional processing gain correction (Skolnik approach)
        processing_gain = np.sqrt(self.time_bandwidth_product)
        compressed_fft *= processing_gain
        
        # Convert back to time domain
        compressed_signal = ifft(compressed_fft)
        
        if plot:
            plt.figure(figsize=(15, 10))
            
            # Time domain plots
            t_plot = np.arange(len(compressed_signal)) / self.fs
            
            plt.subplot(2, 3, 1)
            plt.plot(t_plot*1e6, np.abs(received_signal), 'b-', linewidth=1)
            plt.title('Received Signal')
            plt.xlabel('Time (μs)')
            plt.ylabel('Magnitude')
            plt.grid(True)
            
            plt.subplot(2, 3, 2)
            plt.plot(t_plot*1e6, np.abs(compressed_signal), 'r-', linewidth=1)
            plt.title('Range Compressed Signal')
            plt.xlabel('Time (μs)')
            plt.ylabel('Magnitude')
            plt.grid(True)
            
            # Frequency domain analysis
            freq = fftfreq(len(compressed_signal), 1/self.fs) / 1e6  # MHz
            
            plt.subplot(2, 3, 3)
            plt.plot(freq, np.abs(received_fft), 'g-', linewidth=1)
            plt.title('Received Signal Spectrum')
            plt.xlabel('Frequency (MHz)')
            plt.ylabel('Magnitude')
            plt.grid(True)
            plt.xlim(-self.B/1e6, self.B/1e6)
            
            plt.subplot(2, 3, 4)
            plt.plot(freq, np.abs(ref_fft_conj), 'm-', linewidth=1)
            plt.title('Reference Chirp Spectrum (Conjugated)')
            plt.xlabel('Frequency (MHz)')
            plt.ylabel('Magnitude')
            plt.grid(True)
            plt.xlim(-self.B/1e6, self.B/1e6)
            
            plt.subplot(2, 3, 5)
            plt.plot(freq, np.abs(compressed_fft), 'c-', linewidth=1)
            plt.title('Compressed Signal Spectrum')
            plt.xlabel('Frequency (MHz)')
            plt.ylabel('Magnitude')
            plt.grid(True)
            plt.xlim(-self.B/1e6, self.B/1e6)
            
            # Range profile
            range_axis = t_plot * self.c / 2
            plt.subplot(2, 3, 6)
            plt.plot(range_axis, np.abs(compressed_signal), 'k-', linewidth=1)
            plt.title('Range Profile')
            plt.xlabel('Range (m)')
            plt.ylabel('Magnitude')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('../output/alternative_range_compression.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        return compressed_signal
    
    def calculate_alternative_range_resolution(self):
        """
        Calculate range resolution using alternative formulations
        Cross-validation with different literature sources
        """
        # Skolnik formulation (Radar Handbook)
        res_skolnik = self.c / (2 * self.B)
        
        # Richards formulation (emphasizing pulse compression)
        res_richards = self.c / (2 * self.B)  # Same formula, different derivation
        
        # Omega-K formulation (frequency domain emphasis)
        res_omega_k = self.c / (2 * self.B)  # Fundamental limit remains same
        
        # Alternative: using time-bandwidth product
        res_tb_product = self.c / (2 * self.B * np.sqrt(self.time_bandwidth_product) / self.time_bandwidth_product)
        
        return {
            'skolnik': res_skolnik,
            'richards': res_richards, 
            'omega_k': res_omega_k,
            'tb_product': res_tb_product,
            'primary': res_skolnik  # Use Skolnik as primary
        }
    
    def calculate_processing_gain_alternative(self):
        """
        Calculate processing gain using alternative formulations
        Different from primary model for cross-validation
        """
        # Skolnik approach: Pulse compression ratio
        gain_skolnik = self.time_bandwidth_product
        
        # Richards approach: Signal-to-noise improvement
        gain_richards = self.B * self.Tp
        
        # Omega-K approach: Frequency domain processing gain
        gain_omega_k = np.sqrt(self.time_bandwidth_product)
        
        # Alternative: dB formulation
        gain_db_skolnik = 10 * np.log10(gain_skolnik)
        gain_db_richards = 10 * np.log10(gain_richards)
        gain_db_omega_k = 20 * np.log10(gain_omega_k)
        
        return {
            'linear': {
                'skolnik': gain_skolnik,
                'richards': gain_richards,
                'omega_k': gain_omega_k
            },
            'db': {
                'skolnik': gain_db_skolnik,
                'richards': gain_db_richards,
                'omega_k': gain_db_omega_k
            }
        }
    
    def azimuth_resolution_alternative(self, platform_velocity, synthetic_aperture_length):
        """
        Calculate azimuth resolution using alternative formulations
        """
        # Skolnik formulation
        az_res_skolnik = self.wavelength * 1000 / (2 * synthetic_aperture_length)  # Example range 1km
        
        # Richards formulation (antenna beamwidth approach)
        az_res_richards = self.wavelength / (2 * synthetic_aperture_length / 1000)  # Normalized
        
        # Omega-K formulation (Doppler bandwidth approach)
        doppler_bandwidth = 2 * platform_velocity / self.wavelength
        az_res_omega_k = platform_velocity / doppler_bandwidth
        
        return {
            'skolnik': az_res_skolnik,
            'richards': az_res_richards,
            'omega_k': az_res_omega_k
        }
    
    def cross_validation_metrics(self, target_range=1000):
        """
        Generate comprehensive metrics for cross-validation
        """
        # Range resolution comparison
        range_res = self.calculate_alternative_range_resolution()
        
        # Processing gain comparison
        proc_gain = self.calculate_processing_gain_alternative()
        
        # Generate test signals
        t_rx, received = self.point_target_response_omega_k(R0=target_range, plot=False)
        compressed = self.frequency_domain_range_compression(received, plot=False)
        
        # Peak analysis
        peak_idx = np.argmax(np.abs(compressed))
        peak_magnitude = np.abs(compressed[peak_idx])
        peak_range = (t_rx[peak_idx] * self.c / 2)
        
        # Signal characteristics
        signal_energy = np.sum(np.abs(compressed)**2)
        signal_power = signal_energy / len(compressed)
        
        return {
            'model_type': 'Alternative (Skolnik/Richards/Omega-K)',
            'range_resolution': range_res,
            'processing_gain': proc_gain,
            'target_analysis': {
                'target_range': target_range,
                'detected_range': peak_range,
                'range_error': abs(peak_range - target_range),
                'peak_magnitude': peak_magnitude,
                'signal_energy': signal_energy,
                'signal_power': signal_power
            },
            'system_parameters': {
                'fc': self.fc,
                'B': self.B,
                'Tp': self.Tp,
                'wavelength': self.wavelength,
                'chirp_rate': self.chirp_rate,
                'time_bandwidth_product': self.time_bandwidth_product
            }
        }

def main():
    """Demonstrate Alternative SAR Model"""
    print("ALTERNATIVE SAR MODEL DEMONSTRATION")
    print("="*50)
    print("Based on Skolnik/Richards/Omega-K formulations")
    print()
    
    # Initialize alternative model
    alt_sar = AlternativeSARModel(fc=10e9, B=100e6, Tp=10e-6)
    
    print("\n" + "="*50)
    print("ALTERNATIVE PROCESSING DEMONSTRATION")
    print("="*50)
    
    # Generate and analyze chirp
    print("1. Generating frequency-domain chirp...")
    t, chirp_signal = alt_sar.generate_frequency_domain_chirp(plot=True)
    
    # Point target response
    print("\n2. Generating point target response...")
    t_rx, received = alt_sar.point_target_response_omega_k(R0=1000, plot=True)
    
    # Range compression
    print("\n3. Performing frequency-domain range compression...")
    compressed = alt_sar.frequency_domain_range_compression(received, plot=True)
    
    # Cross-validation metrics
    print("\n4. Generating cross-validation metrics...")
    metrics = alt_sar.cross_validation_metrics(target_range=1000)
    
    print("\n" + "="*50)
    print("ALTERNATIVE MODEL METRICS")
    print("="*50)
    
    print(f"Range Resolution (Skolnik): {metrics['range_resolution']['skolnik']:.3f} m")
    print(f"Processing Gain (Skolnik): {metrics['processing_gain']['db']['skolnik']:.1f} dB")
    print(f"Target Range Error: {metrics['target_analysis']['range_error']:.3f} m")
    print(f"Peak Magnitude: {metrics['target_analysis']['peak_magnitude']:.2e}")
    
    return alt_sar, metrics

if __name__ == "__main__":
    model, results = main()
