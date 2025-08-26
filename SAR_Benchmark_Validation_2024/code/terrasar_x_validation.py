#!/usr/bin/env python3
"""
TerraSAR-X Specific Validation Test
Validates SAR model against TerraSAR-X parameters and specifications
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from sar_model_final import FinalSARModel

def validate_terrasar_x():
    """Validate against TerraSAR-X specifications"""
    print("=" * 60)
    print("TERRASAR-X VALIDATION EXPERIMENT")
    print("=" * 60)
    
    # TerraSAR-X specifications (X-band, high resolution mode)
    fc = 9.65e9      # 9.65 GHz center frequency
    B = 300e6        # 300 MHz bandwidth (high resolution mode)
    Tp = 2.5e-6      # 2.5 microsecond pulse duration
    
    print(f"TerraSAR-X Parameters:")
    print(f"  Frequency: {fc/1e9:.2f} GHz")
    print(f"  Bandwidth: {B/1e6:.0f} MHz")
    print(f"  Pulse duration: {Tp*1e6:.1f} μs")
    print(f"  Expected range resolution: {3e8/(2*B):.2f} m")
    
    # Initialize SAR model
    sar = FinalSARModel(fc=fc, B=B, Tp=Tp)
    
    # Validation tests
    print(f"\n=== MATHEMATICAL VALIDATION ===")
    
    # Range resolution
    theoretical_res = 3e8 / (2 * B)
    actual_res = sar.calculate_range_resolution()
    res_error = abs(theoretical_res - actual_res)
    
    print(f"Range Resolution:")
    print(f"  Theoretical: {theoretical_res:.3f} m")
    print(f"  Model result: {actual_res:.3f} m")
    print(f"  Error: {res_error:.6f} m")
    print(f"  Status: {'✅ PASS' if res_error < 1e-10 else '❌ FAIL'}")
    
    # Wavelength
    theoretical_wavelength = 3e8 / fc
    actual_wavelength = sar.wavelength
    wavelength_error = abs(theoretical_wavelength - actual_wavelength)
    
    print(f"\nWavelength:")
    print(f"  Theoretical: {theoretical_wavelength*100:.2f} cm")
    print(f"  Model result: {actual_wavelength*100:.2f} cm")
    print(f"  Error: {wavelength_error*1000:.3f} mm")
    print(f"  Status: {'✅ PASS' if wavelength_error < 1e-10 else '❌ FAIL'}")
    
    # Time-bandwidth product
    tb_product = sar.get_time_bandwidth_product()
    expected_tb = int(B * Tp)
    
    print(f"\nTime-Bandwidth Product:")
    print(f"  Expected: {expected_tb}")
    print(f"  Model result: {tb_product}")
    print(f"  Status: {'✅ PASS' if tb_product == expected_tb else '❌ FAIL'}")
    
    # Point target testing
    print(f"\n=== POINT TARGET VALIDATION ===")
    test_ranges = [800, 1200, 1600, 2000]  # Typical TerraSAR-X ranges
    
    for R0 in test_ranges:
        # Generate target response
        t, response = sar.point_target_response(R0=R0, plot=False)
        compressed = sar.range_compression(response, plot=False)
        
        # Find peak
        peak_idx = np.argmax(np.abs(compressed))
        
        if peak_idx > 0:
            fs = 200e6
            dt = 1/fs
            peak_time = peak_idx * dt
            detected_range = peak_time * sar.c / 2
            range_error = abs(detected_range - R0)
            
            print(f"Target at {R0}m: Detected {detected_range:.1f}m (error: {range_error:.1f}m) {'✅' if range_error < 10 else '❌'}")
    
    # Generate TerraSAR-X specific visualization
    print(f"\n=== GENERATING TERRASAR-X VISUALIZATIONS ===")
    
    # Generate chirp pulse
    t, pulse = sar.generate_chirp_pulse(plot=False)
    
    plt.figure(figsize=(15, 10))
    
    # Chirp pulse analysis
    plt.subplot(2, 3, 1)
    plt.plot(t*1e6, np.real(pulse))
    plt.title('TerraSAR-X Chirp Pulse - Real Part')
    plt.xlabel('Time (μs)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    plt.subplot(2, 3, 2)
    plt.plot(t*1e6, np.imag(pulse))
    plt.title('TerraSAR-X Chirp Pulse - Imaginary Part')
    plt.xlabel('Time (μs)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    plt.subplot(2, 3, 3)
    plt.plot(t*1e6, np.abs(pulse))
    plt.title('TerraSAR-X Chirp Pulse - Magnitude')
    plt.xlabel('Time (μs)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    # Frequency spectrum
    pulse_fft = np.fft.fft(pulse)
    freqs = np.fft.fftfreq(len(pulse), 1/200e6)
    
    plt.subplot(2, 3, 4)
    plt.plot(freqs/1e6, 20*np.log10(np.abs(pulse_fft) + 1e-10))
    plt.title('TerraSAR-X Chirp Spectrum')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True)
    plt.xlim([-B/2e6, B/2e6])
    
    # Point target response
    t_target, target_response = sar.point_target_response(R0=1000, plot=False)
    compressed = sar.range_compression(target_response, plot=False)
    
    plt.subplot(2, 3, 5)
    plt.plot(t_target*1e6, np.abs(target_response))
    plt.title('TerraSAR-X Target Response')
    plt.xlabel('Time (μs)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    plt.subplot(2, 3, 6)
    range_axis = np.arange(len(compressed)) / 200e6 * 3e8 / 2
    plt.plot(range_axis, 20*np.log10(np.abs(compressed) + 1e-10))
    plt.title('TerraSAR-X Range Compressed')
    plt.xlabel('Range (m)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True)
    plt.xlim([800, 1200])
    
    plt.tight_layout()
    plt.savefig('../output/TerraSAR-X/terrasar_x_validation_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ TerraSAR-X validation complete!")
    print(f"📊 Results saved to: ../output/TerraSAR-X/terrasar_x_validation_results.png")

if __name__ == "__main__":
    validate_terrasar_x()
