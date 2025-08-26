#!/usr/bin/env python3
"""
Algorithm Performance Tests for SAR Model
Tests signal processing algorithms for correctness and performance
"""

import unittest
import numpy as np
import sys
import os
import time

# Add parent directory to path to import SAR modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from sar_basic_model import BasicSARModel
except ImportError:
    print("Warning: Could not import sar_basic_model. Tests will be skipped.")
    BasicSARModel = None

class TestSARAlgorithms(unittest.TestCase):
    """Test SAR signal processing algorithms"""
    
    def setUp(self):
        """Set up test parameters"""
        if BasicSARModel is None:
            self.skipTest("BasicSARModel not available")
        
        self.sar = BasicSARModel(fc=10e9, B=100e6, Tp=10e-6)
        self.fs = 200e6  # Sampling frequency
    
    def test_point_target_detection_accuracy(self):
        """Test point target detection at known range"""
        target_ranges = [500, 1000, 2000, 5000]  # Test multiple ranges
        
        for R0 in target_ranges:
            with self.subTest(range=R0):
                # Generate target response
                t, response = self.sar.point_target_response(R0=R0, plot=False)
                
                # Apply range compression
                compressed = self.sar.range_compression(response, plot=False)
                
                # Find peak
                peak_idx = np.argmax(np.abs(compressed))
                
                # Convert to range
                dt = 1 / self.fs
                peak_time = peak_idx * dt
                detected_range = peak_time * self.sar.c / 2
                
                # Check accuracy (should be within 10% or 50m, whichever is larger)
                range_error = abs(detected_range - R0)
                max_allowed_error = max(R0 * 0.1, 50)
                
                self.assertLess(range_error, max_allowed_error,
                               msg=f"Range detection error {range_error:.1f}m exceeds {max_allowed_error:.1f}m for target at {R0}m")
    
    def test_range_compression_gain(self):
        """Test range compression provides processing gain"""
        R0 = 1000
        t, response = self.sar.point_target_response(R0=R0, plot=False)
        compressed = self.sar.range_compression(response, plot=False)
        
        # Calculate peak powers
        input_peak_power = np.max(np.abs(response)**2)
        output_peak_power = np.max(np.abs(compressed)**2)
        
        # Calculate average noise power (exclude peak region)
        peak_idx = np.argmax(np.abs(compressed))
        noise_region = np.concatenate([
            compressed[:max(0, peak_idx-100)],
            compressed[min(len(compressed), peak_idx+100):]
        ])
        average_noise_power = np.mean(np.abs(noise_region)**2) if len(noise_region) > 0 else 1e-10
        
        # Processing gain should be significant
        processing_gain = output_peak_power / average_noise_power
        
        self.assertGreater(processing_gain, 10,
                          msg=f"Processing gain {processing_gain:.1f} is too low")
    
    def test_matched_filter_properties(self):
        """Test matched filter properties"""
        t, pulse = self.sar.generate_chirp_pulse(fs=self.fs, plot=False)
        
        # Create simple test signal (delayed pulse)
        delay_samples = 100
        test_signal = np.zeros(len(pulse) + delay_samples)
        test_signal[delay_samples:delay_samples+len(pulse)] = pulse
        
        # Apply matched filtering
        compressed = self.sar.range_compression(test_signal, plot=False)
        
        # Peak should be at delay location
        peak_idx = np.argmax(np.abs(compressed))
        expected_peak_idx = delay_samples + len(pulse)//2  # Approximately
        
        peak_error = abs(peak_idx - expected_peak_idx)
        self.assertLess(peak_error, 20,  # Within 20 samples
                       msg=f"Matched filter peak error {peak_error} samples")
    
    def test_multi_target_separation(self):
        """Test ability to separate multiple targets"""
        # Create two targets at different ranges
        R1, R2 = 800, 1200  # 400m separation
        
        t1, response1 = self.sar.point_target_response(R0=R1, plot=False)
        t2, response2 = self.sar.point_target_response(R0=R2, plot=False)
        
        # Combine responses (simulate two targets)
        combined_response = response1 + response2
        
        # Apply range compression
        compressed = self.sar.range_compression(combined_response, plot=False)
        
        # Find peaks
        compressed_abs = np.abs(compressed)
        
        # Find local maxima
        peaks = []
        for i in range(50, len(compressed_abs)-50):
            if (compressed_abs[i] > compressed_abs[i-50:i].max() and 
                compressed_abs[i] > compressed_abs[i+1:i+51].max() and
                compressed_abs[i] > 0.1 * compressed_abs.max()):
                peaks.append(i)
        
        # Should detect at least 2 peaks
        self.assertGreaterEqual(len(peaks), 2,
                               msg=f"Should detect 2 targets, found {len(peaks)} peaks")
    
    def test_chirp_pulse_bandwidth(self):
        """Test chirp pulse frequency content"""
        t, pulse = self.sar.generate_chirp_pulse(fs=self.fs, plot=False)
        
        # Compute FFT
        pulse_fft = np.fft.fft(pulse)
        freqs = np.fft.fftfreq(len(pulse), 1/self.fs)
        
        # Find bandwidth (where power drops to -3dB)
        power_spectrum = np.abs(pulse_fft)**2
        max_power = np.max(power_spectrum)
        half_power = max_power / 2
        
        # Find frequency range where power > half_power
        high_power_indices = np.where(power_spectrum > half_power)[0]
        
        if len(high_power_indices) > 0:
            bandwidth_measured = freqs[high_power_indices[-1]] - freqs[high_power_indices[0]]
            bandwidth_expected = self.sar.B
            
            # Should be within 20% of expected bandwidth
            bandwidth_error = abs(bandwidth_measured - bandwidth_expected) / bandwidth_expected
            self.assertLess(bandwidth_error, 0.2,
                           msg=f"Bandwidth error {bandwidth_error*100:.1f}% too large")

class TestSARPerformance(unittest.TestCase):
    """Test SAR algorithm performance and timing"""
    
    def setUp(self):
        """Set up test parameters"""
        if BasicSARModel is None:
            self.skipTest("BasicSARModel not available")
        self.sar = BasicSARModel()
    
    def test_chirp_generation_speed(self):
        """Test chirp generation performance"""
        start_time = time.time()
        
        # Generate multiple chirps
        for _ in range(10):
            t, pulse = self.sar.generate_chirp_pulse(plot=False)
        
        elapsed_time = time.time() - start_time
        avg_time = elapsed_time / 10
        
        # Should generate chirp in < 0.1 seconds
        self.assertLess(avg_time, 0.1,
                       msg=f"Chirp generation too slow: {avg_time:.3f}s")
    
    def test_range_compression_speed(self):
        """Test range compression performance"""
        # Generate test data
        t, response = self.sar.point_target_response(R0=1000, plot=False)
        
        start_time = time.time()
        
        # Perform multiple compressions
        for _ in range(5):
            compressed = self.sar.range_compression(response, plot=False)
        
        elapsed_time = time.time() - start_time
        avg_time = elapsed_time / 5
        
        # Should compress in < 0.5 seconds
        self.assertLess(avg_time, 0.5,
                       msg=f"Range compression too slow: {avg_time:.3f}s")
    
    def test_memory_usage(self):
        """Test memory usage doesn't grow excessively"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform multiple operations
        for i in range(20):
            t, response = self.sar.point_target_response(R0=1000+i*100, plot=False)
            compressed = self.sar.range_compression(response, plot=False)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be reasonable (< 100 MB)
        self.assertLess(memory_growth, 100,
                       msg=f"Memory growth {memory_growth:.1f}MB too large")

def run_algorithm_tests():
    """Run all algorithm tests and return results"""
    print("=" * 60)
    print("SAR ALGORITHM PERFORMANCE TESTS")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSARAlgorithms))
    suite.addTests(loader.loadTestsFromTestCase(TestSARPerformance))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("ALGORITHM TESTS SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors))/result.testsRun)*100:.1f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_algorithm_tests()
    exit(0 if success else 1)
