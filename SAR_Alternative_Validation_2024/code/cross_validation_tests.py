#!/usr/bin/env python3
"""
Cross-Validation Test Suite
Compares Alternative SAR Model with Primary SAR Model
Tests consistency across different mathematical formulations and approaches
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append('../../SAR_Benchmark_Validation_2024/code')
from sar_model_final import FinalSARModel
from alternative_sar_model import AlternativeSARModel
import json
from datetime import datetime

class CrossValidationTests:
    def __init__(self):
        self.test_results = {}
        self.tolerance_percent = 5.0  # 5% tolerance for cross-validation
        
    def initialize_models(self, fc=10e9, B=100e6, Tp=10e-6):
        """Initialize both SAR models with identical parameters"""
        print("INITIALIZING SAR MODELS FOR CROSS-VALIDATION")
        print("="*60)
        
        print("\n--- PRIMARY MODEL (Cumming/Wong/Soumekh) ---")
        self.primary_model = FinalSARModel(fc=fc, B=B, Tp=Tp)
        
        print("\n--- ALTERNATIVE MODEL (Skolnik/Richards/Omega-K) ---")
        self.alternative_model = AlternativeSARModel(fc=fc, B=B, Tp=Tp)
        
        print("\nModels initialized successfully!")
        return True
    
    def test_range_resolution_consistency(self):
        """Test 1: Range Resolution Formula Consistency"""
        print("\n" + "="*60)
        print("TEST 1: RANGE RESOLUTION CONSISTENCY")
        print("="*60)
        
        # Get range resolution from both models
        primary_res = self.primary_model.calculate_range_resolution()
        alternative_res = self.alternative_model.calculate_alternative_range_resolution()
        
        print(f"Primary Model Resolution: {primary_res:.6f} m")
        print(f"Alternative Model (Skolnik): {alternative_res['skolnik']:.6f} m")
        print(f"Alternative Model (Richards): {alternative_res['richards']:.6f} m")
        print(f"Alternative Model (Omega-K): {alternative_res['omega_k']:.6f} m")
        
        # Calculate differences
        diff_skolnik = abs(primary_res - alternative_res['skolnik']) / primary_res * 100
        diff_richards = abs(primary_res - alternative_res['richards']) / primary_res * 100
        diff_omega_k = abs(primary_res - alternative_res['omega_k']) / primary_res * 100
        
        print(f"\nDifference vs Primary:")
        print(f"Skolnik approach: {diff_skolnik:.8f}%")
        print(f"Richards approach: {diff_richards:.8f}%")
        print(f"Omega-K approach: {diff_omega_k:.8f}%")
        
        # Test passes if all differences are within tolerance
        max_diff = max(diff_skolnik, diff_richards, diff_omega_k)
        test_passed = max_diff < self.tolerance_percent
        
        print(f"\nMaximum Difference: {max_diff:.8f}%")
        print(f"Tolerance: {self.tolerance_percent}%")
        print(f"Test Result: {'PASS' if test_passed else 'FAIL'}")
        
        self.test_results['range_resolution_consistency'] = {
            'primary_resolution': primary_res,
            'alternative_resolutions': alternative_res,
            'differences_percent': {
                'skolnik': diff_skolnik,
                'richards': diff_richards,
                'omega_k': diff_omega_k
            },
            'max_difference_percent': max_diff,
            'passed': test_passed
        }
        
        return test_passed
    
    def test_processing_gain_consistency(self):
        """Test 2: Processing Gain Calculation Consistency"""
        print("\n" + "="*60)
        print("TEST 2: PROCESSING GAIN CONSISTENCY")
        print("="*60)
        
        # Get processing gains from both models
        primary_gain_db = 10 * np.log10(self.primary_model.B * self.primary_model.Tp)
        alternative_gains = self.alternative_model.calculate_processing_gain_alternative()
        
        print(f"Primary Model Gain: {primary_gain_db:.2f} dB")
        print(f"Alternative Model (Skolnik): {alternative_gains['db']['skolnik']:.2f} dB")
        print(f"Alternative Model (Richards): {alternative_gains['db']['richards']:.2f} dB")
        print(f"Alternative Model (Omega-K): {alternative_gains['db']['omega_k']:.2f} dB")
        
        # Calculate differences
        diff_skolnik = abs(primary_gain_db - alternative_gains['db']['skolnik'])
        diff_richards = abs(primary_gain_db - alternative_gains['db']['richards'])
        diff_omega_k = abs(primary_gain_db - alternative_gains['db']['omega_k'])
        
        print(f"\nDifference vs Primary:")
        print(f"Skolnik approach: {diff_skolnik:.2f} dB")
        print(f"Richards approach: {diff_richards:.2f} dB")
        print(f"Omega-K approach: {diff_omega_k:.2f} dB")
        
        # Test passes if differences are reasonable (within 3 dB)
        max_diff = max(diff_skolnik, diff_richards, diff_omega_k)
        test_passed = max_diff < 3.0  # 3 dB tolerance
        
        print(f"\nMaximum Difference: {max_diff:.2f} dB")
        print(f"Tolerance: 3.0 dB")
        print(f"Test Result: {'PASS' if test_passed else 'FAIL'}")
        
        self.test_results['processing_gain_consistency'] = {
            'primary_gain_db': primary_gain_db,
            'alternative_gains_db': alternative_gains['db'],
            'differences_db': {
                'skolnik': diff_skolnik,
                'richards': diff_richards,
                'omega_k': diff_omega_k
            },
            'max_difference_db': max_diff,
            'passed': test_passed
        }
        
        return test_passed
    
    def test_point_target_response_consistency(self):
        """Test 3: Point Target Response Consistency"""
        print("\n" + "="*60)
        print("TEST 3: POINT TARGET RESPONSE CONSISTENCY")
        print("="*60)
        
        # Test parameters
        test_ranges = [500, 1000, 2000]  # meters
        results = []
        
        for R0 in test_ranges:
            print(f"\n--- Testing Range: {R0} m ---")
            
            # Generate responses from both models
            t_primary, response_primary = self.primary_model.point_target_response(R0=R0, plot=False)
            t_alt, response_alt = self.alternative_model.point_target_response_omega_k(R0=R0, plot=False)
            
            # Range compress both responses
            compressed_primary = self.primary_model.range_compression(response_primary, plot=False)
            compressed_alt = self.alternative_model.frequency_domain_range_compression(response_alt, plot=False)
            
            # Find peaks
            peak_idx_primary = np.argmax(np.abs(compressed_primary))
            peak_idx_alt = np.argmax(np.abs(compressed_alt))
            
            # Convert to range
            range_primary = t_primary[peak_idx_primary] * self.primary_model.c / 2
            range_alt = t_alt[peak_idx_alt] * self.alternative_model.c / 2
            
            # Calculate errors
            error_primary = abs(range_primary - R0)
            error_alt = abs(range_alt - R0)
            
            # Cross-model consistency
            range_difference = abs(range_primary - range_alt)
            range_diff_percent = range_difference / R0 * 100
            
            print(f"Primary Model Detected: {range_primary:.3f} m (error: {error_primary:.3f} m)")
            print(f"Alternative Model Detected: {range_alt:.3f} m (error: {error_alt:.3f} m)")
            print(f"Cross-Model Difference: {range_difference:.3f} m ({range_diff_percent:.3f}%)")
            
            # Peak magnitude comparison
            mag_primary = np.abs(compressed_primary[peak_idx_primary])
            mag_alt = np.abs(compressed_alt[peak_idx_alt])
            mag_ratio = mag_alt / mag_primary if mag_primary > 0 else 0
            
            print(f"Peak Magnitude Ratio (Alt/Primary): {mag_ratio:.3f}")
            
            results.append({
                'target_range': R0,
                'primary_detected': range_primary,
                'alternative_detected': range_alt,
                'primary_error': error_primary,
                'alternative_error': error_alt,
                'cross_difference': range_difference,
                'cross_difference_percent': range_diff_percent,
                'magnitude_ratio': mag_ratio
            })
        
        # Overall assessment
        max_cross_diff_percent = max([r['cross_difference_percent'] for r in results])
        test_passed = max_cross_diff_percent < self.tolerance_percent
        
        print(f"\n--- OVERALL ASSESSMENT ---")
        print(f"Maximum Cross-Model Difference: {max_cross_diff_percent:.3f}%")
        print(f"Tolerance: {self.tolerance_percent}%")
        print(f"Test Result: {'PASS' if test_passed else 'FAIL'}")
        
        self.test_results['point_target_consistency'] = {
            'results': results,
            'max_cross_difference_percent': max_cross_diff_percent,
            'passed': test_passed
        }
        
        return test_passed
    
    def test_system_parameter_consistency(self):
        """Test 4: System Parameter Consistency"""
        print("\n" + "="*60)
        print("TEST 4: SYSTEM PARAMETER CONSISTENCY")
        print("="*60)
        
        # Compare fundamental parameters
        params_primary = {
            'fc': self.primary_model.fc,
            'B': self.primary_model.B,
            'Tp': self.primary_model.Tp,
            'wavelength': self.primary_model.wavelength,
            'Kr': self.primary_model.Kr
        }
        
        params_alt = {
            'fc': self.alternative_model.fc,
            'B': self.alternative_model.B,
            'Tp': self.alternative_model.Tp,
            'wavelength': self.alternative_model.wavelength,
            'Kr': self.alternative_model.chirp_rate
        }
        
        print("Parameter Comparison:")
        print(f"{'Parameter':<12} {'Primary':<15} {'Alternative':<15} {'Difference':<12}")
        print("-" * 60)
        
        all_consistent = True
        max_param_diff = 0
        
        for param in params_primary:
            val_primary = params_primary[param]
            val_alt = params_alt[param]
            diff_percent = abs(val_primary - val_alt) / val_primary * 100 if val_primary != 0 else 0
            max_param_diff = max(max_param_diff, diff_percent)
            
            if diff_percent > 0.001:  # 0.001% tolerance for parameters
                all_consistent = False
            
            print(f"{param:<12} {val_primary:<15.3e} {val_alt:<15.3e} {diff_percent:<12.6f}%")
        
        print(f"\nMaximum Parameter Difference: {max_param_diff:.8f}%")
        print(f"Test Result: {'PASS' if all_consistent else 'FAIL'}")
        
        self.test_results['parameter_consistency'] = {
            'primary_parameters': params_primary,
            'alternative_parameters': params_alt,
            'max_difference_percent': max_param_diff,
            'passed': all_consistent
        }
        
        return all_consistent
    
    def generate_cross_validation_plots(self):
        """Generate comparative plots between models"""
        print("\n" + "="*60)
        print("GENERATING CROSS-VALIDATION PLOTS")
        print("="*60)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('SAR Model Cross-Validation Analysis', fontsize=16)
        
        # Test parameters
        R0 = 1000  # 1 km target
        
        # 1. Chirp comparison
        t_primary, chirp_primary = self.primary_model.generate_chirp_pulse()
        t_alt, chirp_alt = self.alternative_model.generate_frequency_domain_chirp()
        
        axes[0,0].plot(t_primary*1e6, np.real(chirp_primary), 'b-', label='Primary (Time Domain)', linewidth=1)
        axes[0,0].plot(t_alt*1e6, np.real(chirp_alt), 'r--', label='Alternative (Freq Domain)', linewidth=1)
        axes[0,0].set_title('Chirp Signal Comparison - Real Part')
        axes[0,0].set_xlabel('Time (μs)')
        axes[0,0].set_ylabel('Amplitude')
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        # 2. Point target responses
        t_rx_primary, response_primary = self.primary_model.point_target_response(R0=R0, plot=False)
        t_rx_alt, response_alt = self.alternative_model.point_target_response_omega_k(R0=R0, plot=False)
        
        axes[0,1].plot(t_rx_primary*1e6, np.abs(response_primary), 'b-', label='Primary Model', linewidth=1)
        axes[0,1].plot(t_rx_alt*1e6, np.abs(response_alt), 'r--', label='Alternative Model', linewidth=1)
        axes[0,1].set_title(f'Point Target Response (R0={R0}m)')
        axes[0,1].set_xlabel('Time (μs)')
        axes[0,1].set_ylabel('Magnitude')
        axes[0,1].legend()
        axes[0,1].grid(True)
        
        # 3. Range compressed signals
        compressed_primary = self.primary_model.range_compression(response_primary, plot=False)
        compressed_alt = self.alternative_model.frequency_domain_range_compression(response_alt, plot=False)
        
        range_axis_primary = t_rx_primary * self.primary_model.c / 2
        range_axis_alt = t_rx_alt * self.alternative_model.c / 2
        
        axes[0,2].plot(range_axis_primary, np.abs(compressed_primary), 'b-', label='Primary Model', linewidth=1)
        axes[0,2].plot(range_axis_alt, np.abs(compressed_alt), 'r--', label='Alternative Model', linewidth=1)
        axes[0,2].set_title('Range Compressed Signals')
        axes[0,2].set_xlabel('Range (m)')
        axes[0,2].set_ylabel('Magnitude')
        axes[0,2].legend()
        axes[0,2].grid(True)
        axes[0,2].set_xlim(R0-50, R0+50)
        
        # 4. Range resolution comparison
        bandwidths = np.array([25, 50, 100, 200, 400]) * 1e6  # MHz to Hz
        res_primary = []
        res_alt = []
        
        for B in bandwidths:
            res_primary.append(self.primary_model.c / (2 * B))
            temp_alt = AlternativeSARModel(fc=10e9, B=B, Tp=10e-6)
            res_alt.append(temp_alt.calculate_alternative_range_resolution()['primary'])
        
        axes[1,0].plot(bandwidths/1e6, res_primary, 'bo-', label='Primary Model', linewidth=2)
        axes[1,0].plot(bandwidths/1e6, res_alt, 'rs--', label='Alternative Model', linewidth=2)
        axes[1,0].set_title('Range Resolution vs Bandwidth')
        axes[1,0].set_xlabel('Bandwidth (MHz)')
        axes[1,0].set_ylabel('Range Resolution (m)')
        axes[1,0].legend()
        axes[1,0].grid(True)
        axes[1,0].set_yscale('log')
        
        # 5. Processing gain comparison
        proc_gain_primary = []
        proc_gain_alt = []
        
        for B in bandwidths:
            proc_gain_primary.append(10 * np.log10(B * 10e-6))  # Fixed Tp
            temp_alt = AlternativeSARModel(fc=10e9, B=B, Tp=10e-6)
            proc_gain_alt.append(temp_alt.calculate_processing_gain_alternative()['db']['skolnik'])
        
        axes[1,1].plot(bandwidths/1e6, proc_gain_primary, 'bo-', label='Primary Model', linewidth=2)
        axes[1,1].plot(bandwidths/1e6, proc_gain_alt, 'rs--', label='Alternative Model', linewidth=2)
        axes[1,1].set_title('Processing Gain vs Bandwidth')
        axes[1,1].set_xlabel('Bandwidth (MHz)')
        axes[1,1].set_ylabel('Processing Gain (dB)')
        axes[1,1].legend()
        axes[1,1].grid(True)
        
        # 6. Cross-validation summary
        if hasattr(self, 'test_results') and self.test_results:
            test_names = []
            test_status = []
            
            for test_name, result in self.test_results.items():
                if 'passed' in result:
                    test_names.append(test_name.replace('_', '\n'))
                    test_status.append(1 if result['passed'] else 0)
            
            colors = ['green' if status else 'red' for status in test_status]
            axes[1,2].bar(range(len(test_names)), test_status, color=colors, alpha=0.7)
            axes[1,2].set_title('Cross-Validation Test Results')
            axes[1,2].set_xlabel('Test')
            axes[1,2].set_ylabel('Pass (1) / Fail (0)')
            axes[1,2].set_xticks(range(len(test_names)))
            axes[1,2].set_xticklabels(test_names, rotation=45, fontsize=8)
            axes[1,2].set_ylim(-0.1, 1.1)
            axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../output/cross_validation_analysis.png', dpi=300, bbox_inches='tight')
        print("Cross-validation plots saved to: ../output/cross_validation_analysis.png")
        plt.close()
    
    def run_cross_validation_suite(self):
        """Run complete cross-validation test suite"""
        print("SAR MODEL CROSS-VALIDATION TEST SUITE")
        print("="*60)
        print("Comparing Primary vs Alternative SAR implementations")
        print()
        
        # Initialize models
        self.initialize_models(fc=10e9, B=100e6, Tp=10e-6)
        
        # Run all tests
        test_methods = [
            self.test_range_resolution_consistency,
            self.test_processing_gain_consistency,
            self.test_point_target_response_consistency,
            self.test_system_parameter_consistency
        ]
        
        passed_tests = 0
        total_tests = len(test_methods)
        
        for test_method in test_methods:
            try:
                if test_method():
                    passed_tests += 1
            except Exception as e:
                print(f"Test {test_method.__name__} failed with error: {e}")
        
        # Generate plots
        self.generate_cross_validation_plots()
        
        # Final summary
        print("\n" + "="*60)
        print("CROSS-VALIDATION TEST SUMMARY")
        print("="*60)
        print(f"Tests Passed: {passed_tests}/{total_tests}")
        print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
        
        overall_passed = passed_tests == total_tests
        print(f"Overall Result: {'PASS' if overall_passed else 'FAIL'}")
        
        # Add summary to results
        self.test_results['summary'] = {
            'tests_passed': passed_tests,
            'total_tests': total_tests,
            'success_rate': passed_tests/total_tests*100,
            'overall_passed': overall_passed,
            'timestamp': datetime.now().isoformat()
        }
        
        return overall_passed, self.test_results

def main():
    """Main execution function"""
    # Create output directory
    os.makedirs('../output', exist_ok=True)
    
    # Run cross-validation tests
    tester = CrossValidationTests()
    success, results = tester.run_cross_validation_suite()
    
    # Save results
    def convert_numpy_types(obj):
        """Convert numpy types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    with open('../output/cross_validation_results.json', 'w') as f:
        json.dump(convert_numpy_types(results), f, indent=2)
    
    print(f"\nResults saved to: ../output/cross_validation_results.json")
    return success

if __name__ == "__main__":
    success = main()
