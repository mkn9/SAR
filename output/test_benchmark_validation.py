#!/usr/bin/env python3
"""
SAR Model Benchmark Validation
Compare our SAR model results against known theoretical values and published benchmarks
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import SAR modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from sar_basic_model import BasicSARModel
except ImportError:
    print("Warning: Could not import sar_basic_model. Validation will be skipped.")
    BasicSARModel = None

class SARBenchmarkValidator:
    """Validate SAR model against known benchmarks and theoretical results"""
    
    def __init__(self):
        """Initialize validator with standard SAR parameters"""
        self.results = {}
        
    def validate_standard_x_band_parameters(self):
        """Validate against standard X-band SAR parameters (similar to TerraSAR-X)"""
        print("=" * 60)
        print("VALIDATION AGAINST STANDARD X-BAND SAR PARAMETERS")
        print("=" * 60)
        
        # Standard X-band parameters (similar to TerraSAR-X, COSMO-SkyMed)
        fc = 9.65e9      # 9.65 GHz (X-band center frequency)
        B = 300e6        # 300 MHz bandwidth (high resolution mode)
        Tp = 2.5e-6      # 2.5 microsecond pulse
        
        sar = BasicSARModel(fc=fc, B=B, Tp=Tp)
        
        # Expected theoretical values
        expected_range_res = 3e8 / (2 * B)  # Should be ~0.5m
        expected_wavelength = 3e8 / fc       # Should be ~3.1cm
        expected_chirp_rate = B / Tp         # Should be 1.2e14 Hz/s
        
        # Our model results
        actual_range_res = sar.calculate_range_resolution()
        actual_wavelength = sar.wavelength
        actual_chirp_rate = sar.Kr
        
        print(f"Range Resolution:")
        print(f"  Expected: {expected_range_res:.3f} m")
        print(f"  Our model: {actual_range_res:.3f} m")
        print(f"  Error: {abs(expected_range_res - actual_range_res):.6f} m")
        print(f"  Status: {'✅ PASS' if abs(expected_range_res - actual_range_res) < 1e-10 else '❌ FAIL'}")
        
        print(f"\nWavelength:")
        print(f"  Expected: {expected_wavelength*100:.2f} cm")
        print(f"  Our model: {actual_wavelength*100:.2f} cm")
        print(f"  Error: {abs(expected_wavelength - actual_wavelength)*1000:.3f} mm")
        print(f"  Status: {'✅ PASS' if abs(expected_wavelength - actual_wavelength) < 1e-10 else '❌ FAIL'}")
        
        print(f"\nChirp Rate:")
        print(f"  Expected: {expected_chirp_rate:.2e} Hz/s")
        print(f"  Our model: {actual_chirp_rate:.2e} Hz/s")
        print(f"  Error: {abs(expected_chirp_rate - actual_chirp_rate):.2e} Hz/s")
        print(f"  Status: {'✅ PASS' if abs(expected_chirp_rate - actual_chirp_rate) < 1e6 else '❌ FAIL'}")
        
        self.results['x_band_validation'] = {
            'range_res_error': abs(expected_range_res - actual_range_res),
            'wavelength_error': abs(expected_wavelength - actual_wavelength),
            'chirp_rate_error': abs(expected_chirp_rate - actual_chirp_rate)
        }
        
        return all([
            abs(expected_range_res - actual_range_res) < 1e-10,
            abs(expected_wavelength - actual_wavelength) < 1e-10,
            abs(expected_chirp_rate - actual_chirp_rate) < 1e6
        ])
    
    def validate_point_target_response_benchmark(self):
        """Validate point target response against theoretical expectations"""
        print("\n" + "=" * 60)
        print("POINT TARGET RESPONSE VALIDATION")
        print("=" * 60)
        
        sar = BasicSARModel(fc=10e9, B=100e6, Tp=10e-6)
        
        # Test multiple target ranges
        test_ranges = [500, 1000, 2000, 5000]  # meters
        range_errors = []
        
        for R0 in test_ranges:
            # Generate target response
            t, response = sar.point_target_response(R0=R0, plot=False)
            compressed = sar.range_compression(response, plot=False)
            
            # Find peak location
            peak_idx = np.argmax(np.abs(compressed))
            fs = 200e6  # Sampling frequency
            dt = 1/fs
            peak_time = peak_idx * dt
            detected_range = peak_time * sar.c / 2
            
            range_error = abs(detected_range - R0)
            range_errors.append(range_error)
            
            print(f"Target at {R0}m:")
            print(f"  Detected: {detected_range:.1f}m")
            print(f"  Error: {range_error:.1f}m")
            print(f"  Status: {'✅ PASS' if range_error < 50 else '❌ FAIL'}")
        
        avg_error = np.mean(range_errors)
        max_error = np.max(range_errors)
        
        print(f"\nSummary:")
        print(f"  Average error: {avg_error:.1f}m")
        print(f"  Maximum error: {max_error:.1f}m")
        print(f"  Overall: {'✅ PASS' if max_error < 50 else '❌ FAIL'}")
        
        self.results['point_target_validation'] = {
            'avg_error': avg_error,
            'max_error': max_error,
            'individual_errors': range_errors
        }
        
        return max_error < 50
    
    def validate_processing_gain_benchmark(self):
        """Validate range compression processing gain"""
        print("\n" + "=" * 60)
        print("PROCESSING GAIN VALIDATION")
        print("=" * 60)
        
        sar = BasicSARModel(fc=10e9, B=100e6, Tp=10e-6)
        
        # Generate target response
        R0 = 1000
        t, response = sar.point_target_response(R0=R0, plot=False)
        compressed = sar.range_compression(response, plot=False)
        
        # Calculate theoretical processing gain
        # For matched filter: Gain ≈ Time-Bandwidth product = B * Tp
        theoretical_gain_db = 10 * np.log10(sar.B * sar.Tp)
        
        # Calculate actual processing gain
        peak_power = np.max(np.abs(compressed)**2)
        
        # Estimate noise power (exclude peak region)
        peak_idx = np.argmax(np.abs(compressed))
        noise_region = np.concatenate([
            compressed[:max(0, peak_idx-200)],
            compressed[min(len(compressed), peak_idx+200):]
        ])
        noise_power = np.mean(np.abs(noise_region)**2) if len(noise_region) > 0 else 1e-10
        
        actual_gain_linear = peak_power / noise_power
        actual_gain_db = 10 * np.log10(actual_gain_linear)
        
        gain_error_db = abs(theoretical_gain_db - actual_gain_db)
        
        print(f"Processing Gain:")
        print(f"  Theoretical: {theoretical_gain_db:.1f} dB")
        print(f"  Our model: {actual_gain_db:.1f} dB")
        print(f"  Error: {gain_error_db:.1f} dB")
        print(f"  Status: {'✅ PASS' if gain_error_db < 5 else '❌ FAIL'}")
        
        # Additional check: Time-Bandwidth product
        tb_product = sar.B * sar.Tp
        expected_tb = 1000  # For our parameters: 100MHz * 10μs = 1000
        
        print(f"\nTime-Bandwidth Product:")
        print(f"  Expected: {expected_tb}")
        print(f"  Our model: {tb_product}")
        print(f"  Status: {'✅ PASS' if tb_product == expected_tb else '❌ FAIL'}")
        
        self.results['processing_gain_validation'] = {
            'theoretical_gain_db': theoretical_gain_db,
            'actual_gain_db': actual_gain_db,
            'gain_error_db': gain_error_db,
            'tb_product': tb_product
        }
        
        return gain_error_db < 5 and tb_product == expected_tb
    
    def validate_against_literature_values(self):
        """Validate against published SAR literature values"""
        print("\n" + "=" * 60)
        print("LITERATURE VALUES VALIDATION")
        print("=" * 60)
        
        # Common SAR system comparisons from literature
        systems = {
            'TerraSAR-X': {
                'fc': 9.65e9,
                'B': 300e6,
                'expected_res': 0.5,  # meters
                'wavelength': 0.031   # meters
            },
            'RADARSAT-2': {
                'fc': 5.405e9,
                'B': 100e6,
                'expected_res': 1.5,  # meters  
                'wavelength': 0.0555  # meters
            },
            'Sentinel-1': {
                'fc': 5.405e9,
                'B': 56.5e6,
                'expected_res': 2.7,  # meters
                'wavelength': 0.0555  # meters
            }
        }
        
        all_passed = True
        
        for system_name, params in systems.items():
            print(f"\n{system_name} Comparison:")
            
            sar = BasicSARModel(fc=params['fc'], B=params['B'])
            
            # Range resolution comparison
            our_res = sar.calculate_range_resolution()
            res_error = abs(our_res - params['expected_res'])
            
            print(f"  Range Resolution: {our_res:.1f}m (expected {params['expected_res']}m)")
            print(f"  Error: {res_error:.3f}m")
            
            # Wavelength comparison
            wavelength_error = abs(sar.wavelength - params['wavelength'])
            print(f"  Wavelength: {sar.wavelength:.4f}m (expected {params['wavelength']}m)")
            print(f"  Error: {wavelength_error:.6f}m")
            
            system_pass = res_error < 0.1 and wavelength_error < 0.001
            print(f"  Status: {'✅ PASS' if system_pass else '❌ FAIL'}")
            
            all_passed = all_passed and system_pass
        
        self.results['literature_validation'] = systems
        return all_passed
    
    def generate_validation_report(self):
        """Generate comprehensive validation report"""
        print("\n" + "=" * 60)
        print("BENCHMARK VALIDATION SUMMARY")
        print("=" * 60)
        
        total_tests = 0
        passed_tests = 0
        
        # Run all validations
        validations = [
            ("X-Band Parameters", self.validate_standard_x_band_parameters),
            ("Point Target Response", self.validate_point_target_response_benchmark),
            ("Processing Gain", self.validate_processing_gain_benchmark),
            ("Literature Values", self.validate_against_literature_values)
        ]
        
        results_summary = []
        
        for test_name, test_func in validations:
            try:
                result = test_func()
                total_tests += 1
                if result:
                    passed_tests += 1
                results_summary.append((test_name, result))
            except Exception as e:
                print(f"Error in {test_name}: {e}")
                results_summary.append((test_name, False))
                total_tests += 1
        
        print("\n" + "=" * 60)
        print("FINAL VALIDATION RESULTS")
        print("=" * 60)
        
        for test_name, passed in results_summary:
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{test_name:.<30} {status}")
        
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"\nOverall Results:")
        print(f"Tests passed: {passed_tests}/{total_tests}")
        print(f"Success rate: {success_rate:.1f}%")
        
        if success_rate >= 75:
            print("🎉 SAR MODEL VALIDATION: EXCELLENT")
        elif success_rate >= 50:
            print("⚠️  SAR MODEL VALIDATION: GOOD")
        else:
            print("❌ SAR MODEL VALIDATION: NEEDS IMPROVEMENT")
        
        return success_rate >= 75

def main():
    """Run benchmark validation"""
    if BasicSARModel is None:
        print("❌ Cannot run validation: SAR model not available")
        return False
    
    validator = SARBenchmarkValidator()
    success = validator.generate_validation_report()
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
