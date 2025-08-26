#!/usr/bin/env python3
"""
Mathematical Validation Tests for SAR Model
Tests core mathematical functions and equations for accuracy
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path to import SAR modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from sar_basic_model import BasicSARModel
except ImportError:
    print("Warning: Could not import sar_basic_model. Tests will be skipped.")
    BasicSARModel = None

class TestSARMathematics(unittest.TestCase):
    """Test mathematical accuracy of SAR model"""
    
    def setUp(self):
        """Set up test parameters"""
        if BasicSARModel is None:
            self.skipTest("BasicSARModel not available")
        
        # Standard test parameters
        self.fc = 10e9      # 10 GHz X-band
        self.B = 100e6      # 100 MHz bandwidth
        self.Tp = 10e-6     # 10 microsecond pulse
        self.c = 3e8        # Speed of light
        
        self.sar = BasicSARModel(fc=self.fc, B=self.B, Tp=self.Tp, c=self.c)
    
    def test_range_resolution_calculation(self):
        """Test range resolution formula: ΔR = c/(2*B)"""
        expected_resolution = self.c / (2 * self.B)
        calculated_resolution = self.sar.calculate_range_resolution()
        
        self.assertAlmostEqual(expected_resolution, calculated_resolution, places=10,
                              msg=f"Range resolution mismatch: expected {expected_resolution}, got {calculated_resolution}")
        
        # Should be 1.5 meters for 100 MHz bandwidth
        self.assertAlmostEqual(calculated_resolution, 1.5, places=1,
                              msg=f"Range resolution should be 1.5m, got {calculated_resolution}")
    
    def test_wavelength_calculation(self):
        """Test wavelength formula: λ = c/fc"""
        expected_wavelength = self.c / self.fc
        actual_wavelength = self.sar.wavelength
        
        self.assertAlmostEqual(expected_wavelength, actual_wavelength, places=10,
                              msg=f"Wavelength mismatch: expected {expected_wavelength}, got {actual_wavelength}")
        
        # Should be 3 cm for 10 GHz
        self.assertAlmostEqual(actual_wavelength, 0.03, places=3,
                              msg=f"Wavelength should be 0.03m, got {actual_wavelength}")
    
    def test_chirp_rate_calculation(self):
        """Test chirp rate formula: Kr = B/Tp"""
        expected_Kr = self.B / self.Tp
        actual_Kr = self.sar.Kr
        
        self.assertAlmostEqual(expected_Kr, actual_Kr, places=6,
                              msg=f"Chirp rate mismatch: expected {expected_Kr}, got {actual_Kr}")
        
        # Should be 10^13 Hz/s for our parameters
        self.assertEqual(actual_Kr, 1e13,
                        msg=f"Chirp rate should be 1e13 Hz/s, got {actual_Kr}")
    
    def test_chirp_pulse_properties(self):
        """Test LFM chirp pulse generation properties"""
        fs = 200e6
        t, pulse = self.sar.generate_chirp_pulse(fs=fs, plot=False)
        
        # Test pulse length
        expected_samples = int(fs * self.Tp)
        self.assertEqual(len(pulse), expected_samples,
                        msg=f"Pulse length mismatch: expected {expected_samples}, got {len(pulse)}")
        
        # Test pulse amplitude (should be normalized)
        max_amplitude = np.max(np.abs(pulse))
        self.assertAlmostEqual(max_amplitude, 1.0, places=10,
                              msg=f"Pulse amplitude should be 1.0, got {max_amplitude}")
        
        # Test pulse is complex
        self.assertTrue(np.iscomplexobj(pulse),
                       msg="Pulse should be complex-valued")
        
        # Test no NaN or infinite values
        self.assertTrue(np.all(np.isfinite(pulse)),
                       msg="Pulse contains NaN or infinite values")
    
    def test_phase_progression(self):
        """Test LFM chirp phase progression"""
        fs = 200e6
        t, pulse = self.sar.generate_chirp_pulse(fs=fs, plot=False)
        
        # Extract phase
        phase = np.angle(pulse)
        
        # Phase should be quadratic in time (LFM property)
        # Test by checking phase derivative increases linearly
        phase_unwrapped = np.unwrap(phase)
        phase_derivative = np.diff(phase_unwrapped)
        
        # Should have increasing frequency (positive slope in phase derivative)
        self.assertTrue(np.mean(phase_derivative[1:] - phase_derivative[:-1]) > 0,
                       msg="Phase derivative should increase (chirp property)")
    
    def test_unambiguous_range_calculation(self):
        """Test unambiguous range formula: Ru = c/(2*PRF)"""
        PRF = 1000  # 1 kHz PRF
        expected_range = self.c / (2 * PRF)
        calculated_range = self.sar.calculate_unambiguous_range(PRF)
        
        self.assertAlmostEqual(expected_range, calculated_range, places=6,
                              msg=f"Unambiguous range mismatch: expected {expected_range}, got {calculated_range}")
        
        # Should be 150 km for 1 kHz PRF
        self.assertAlmostEqual(calculated_range, 150000, places=0,
                              msg=f"Unambiguous range should be 150000m, got {calculated_range}")
    
    def test_target_phase_calculation(self):
        """Test target phase formula: φ = -4πR/λ"""
        R0 = 1000  # 1 km range
        t, response = self.sar.point_target_response(R0=R0, plot=False)
        
        # Find peak in response
        peak_idx = np.argmax(np.abs(response))
        peak_phase = np.angle(response[peak_idx])
        
        # Calculate expected phase
        expected_phase = -4 * np.pi * R0 / self.sar.wavelength
        
        # Phase wraps around, so compare modulo 2π
        phase_diff = np.abs(np.angle(np.exp(1j * (peak_phase - expected_phase))))
        
        self.assertLess(phase_diff, 0.1,  # Within 0.1 radians
                       msg=f"Target phase error too large: {phase_diff} radians")

class TestSARNumericalStability(unittest.TestCase):
    """Test numerical stability and edge cases"""
    
    def setUp(self):
        """Set up test parameters"""
        if BasicSARModel is None:
            self.skipTest("BasicSARModel not available")
        self.sar = BasicSARModel()
    
    def test_zero_range_handling(self):
        """Test handling of zero range (should not crash)"""
        try:
            t, response = self.sar.point_target_response(R0=0, plot=False)
            # Should not crash, response should be valid
            self.assertTrue(len(response) > 0, msg="Zero range should produce valid response")
        except Exception as e:
            self.fail(f"Zero range handling failed: {e}")
    
    def test_large_range_handling(self):
        """Test handling of very large ranges"""
        large_range = 1e6  # 1000 km
        try:
            t, response = self.sar.point_target_response(R0=large_range, plot=False)
            self.assertTrue(len(response) > 0, msg="Large range should produce valid response")
        except Exception as e:
            self.fail(f"Large range handling failed: {e}")
    
    def test_small_bandwidth_handling(self):
        """Test handling of small bandwidth"""
        try:
            sar_small_b = BasicSARModel(B=1e6)  # 1 MHz bandwidth
            resolution = sar_small_b.calculate_range_resolution()
            self.assertGreater(resolution, 0, msg="Small bandwidth should give positive resolution")
        except Exception as e:
            self.fail(f"Small bandwidth handling failed: {e}")

def run_math_tests():
    """Run all mathematical tests and return results"""
    print("=" * 60)
    print("SAR MATHEMATICAL VALIDATION TESTS")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSARMathematics))
    suite.addTests(loader.loadTestsFromTestCase(TestSARNumericalStability))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("MATHEMATICAL TESTS SUMMARY")
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
    success = run_math_tests()
    exit(0 if success else 1)
