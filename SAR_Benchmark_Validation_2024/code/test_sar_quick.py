#!/usr/bin/env python3
"""
Quick SAR Model Unit Tests
Fast validation of core SAR functionality for pre-commit testing
Execution time: < 30 seconds
"""

import sys
import time
import numpy as np
from sar_basic_model import BasicSARModel

class SARQuickTester:
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.start_time = time.time()
        
    def assert_test(self, condition, test_name, details=""):
        """Custom assertion with reporting"""
        self.tests_run += 1
        if condition:
            self.tests_passed += 1
            print(f"✅ {test_name}")
            if details:
                print(f"   {details}")
        else:
            print(f"❌ {test_name}")
            if details:
                print(f"   FAILED: {details}")
            
    def test_mathematical_accuracy(self):
        """Test core mathematical functions"""
        print("\n1. Mathematical Accuracy Tests")
        print("-" * 40)
        
        # Test range resolution calculation
        sar = BasicSARModel(fc=10e9, B=100e6, Tp=10e-6)
        theoretical_res = sar.c / (2 * sar.B)
        calculated_res = sar.calculate_range_resolution()
        
        self.assert_test(
            abs(theoretical_res - calculated_res) < 1e-10,
            "Range resolution calculation",
            f"Expected: {theoretical_res:.3f}m, Got: {calculated_res:.3f}m"
        )
        
        # Test wavelength calculation
        expected_wavelength = sar.c / sar.fc
        self.assert_test(
            abs(sar.wavelength - expected_wavelength) < 1e-10,
            "Wavelength calculation",
            f"Expected: {expected_wavelength:.4f}m, Got: {sar.wavelength:.4f}m"
        )
        
        # Test chirp rate calculation
        expected_Kr = sar.B / sar.Tp
        self.assert_test(
            abs(sar.Kr - expected_Kr) < 1e-6,
            "Chirp rate calculation", 
            f"Expected: {expected_Kr:.0f} Hz/s, Got: {sar.Kr:.0f} Hz/s"
        )
        
    def test_signal_generation(self):
        """Test signal generation functions"""
        print("\n2. Signal Generation Tests")
        print("-" * 40)
        
        sar = BasicSARModel()
        
        # Test chirp pulse generation
        t, pulse = sar.generate_chirp_pulse(fs=200e6, plot=False)
        
        self.assert_test(
            len(pulse) > 0,
            "Chirp pulse length",
            f"Generated {len(pulse)} samples"
        )
        
        self.assert_test(
            np.max(np.abs(pulse)) > 0.9,
            "Chirp pulse amplitude",
            f"Max amplitude: {np.max(np.abs(pulse)):.3f}"
        )
        
        self.assert_test(
            np.all(np.isfinite(pulse)),
            "Chirp pulse finite values",
            "All values are finite"
        )
        
        # Test point target response
        t_target, response = sar.point_target_response(R0=1000, plot=False)
        
        self.assert_test(
            len(response) > len(pulse),
            "Target response length",
            f"Response: {len(response)}, Pulse: {len(pulse)}"
        )
        
        self.assert_test(
            np.max(np.abs(response)) > 0,
            "Target response amplitude",
            f"Max amplitude: {np.max(np.abs(response)):.6f}"
        )
        
    def test_processing_algorithms(self):
        """Test signal processing algorithms"""
        print("\n3. Processing Algorithm Tests")
        print("-" * 40)
        
        sar = BasicSARModel()
        
        # Generate test data
        t_target, response = sar.point_target_response(R0=1000, plot=False)
        
        # Test range compression
        compressed = sar.range_compression(response, plot=False)
        
        self.assert_test(
            len(compressed) == len(response),
            "Range compression output length",
            f"Input: {len(response)}, Output: {len(compressed)}"
        )
        
        # Find peak in compressed signal
        peak_idx = np.argmax(np.abs(compressed))
        peak_power = np.abs(compressed[peak_idx])**2
        
        self.assert_test(
            peak_power > 0,
            "Range compression produces peak",
            f"Peak power: {peak_power:.3f}"
        )
        
        # Test compression gain (should be significant)
        input_power = np.mean(np.abs(response)**2)
        compression_gain = peak_power / input_power if input_power > 0 else 0
        
        self.assert_test(
            compression_gain > 10,  # Should have significant gain
            "Range compression gain",
            f"Gain: {compression_gain:.1f}"
        )
        
    def test_system_integration(self):
        """Test complete system integration"""
        print("\n4. System Integration Tests")
        print("-" * 40)
        
        try:
            # Test complete workflow without plotting
            sar = BasicSARModel(fc=10e9, B=100e6, Tp=10e-6)
            
            # Generate chirp
            t, pulse = sar.generate_chirp_pulse(plot=False)
            
            # Simulate target
            t_target, response = sar.point_target_response(R0=1000, plot=False)
            
            # Compress
            compressed = sar.range_compression(response, plot=False)
            
            # Find peak and validate position
            peak_idx = np.argmax(np.abs(compressed))
            fs = 200e6  # Sampling frequency
            dt = 1/fs
            peak_time = peak_idx * dt
            peak_range = peak_time * sar.c / 2
            
            range_error = abs(peak_range - 1000)  # Should be near 1000m
            
            self.assert_test(
                range_error < 50,  # Within 50m accuracy
                "Target range accuracy",
                f"Target at 1000m, detected at {peak_range:.1f}m (error: {range_error:.1f}m)"
            )
            
            self.assert_test(
                True,
                "Complete workflow execution",
                "All processing steps completed successfully"
            )
            
        except Exception as e:
            self.assert_test(
                False,
                "Complete workflow execution",
                f"Exception: {str(e)}"
            )
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("=" * 60)
        print("SAR MODEL QUICK UNIT TESTS")
        print("=" * 60)
        
        self.test_mathematical_accuracy()
        self.test_signal_generation()
        self.test_processing_algorithms()
        self.test_system_integration()
        
        # Summary
        elapsed_time = time.time() - self.start_time
        pass_rate = (self.tests_passed / self.tests_run) * 100 if self.tests_run > 0 else 0
        
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Tests run: {self.tests_run}")
        print(f"Tests passed: {self.tests_passed}")
        print(f"Tests failed: {self.tests_run - self.tests_passed}")
        print(f"Pass rate: {pass_rate:.1f}%")
        print(f"Execution time: {elapsed_time:.2f} seconds")
        
        if pass_rate >= 95:
            print("✅ TESTS PASSED - System ready for use")
            return True
        else:
            print("❌ TESTS FAILED - System needs attention")
            return False

def main():
    """Main test execution"""
    tester = SARQuickTester()
    success = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
