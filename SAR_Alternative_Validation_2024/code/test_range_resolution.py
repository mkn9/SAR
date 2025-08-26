#!/usr/bin/env python3
"""
Range Resolution Validation Tests
Tests range resolution capabilities using multiple validation approaches
Based on fundamental radar theory: ΔR = c/(2*B)
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append('../../SAR_Benchmark_Validation_2024/code')
from sar_model_final import FinalSARModel
import unittest

class RangeResolutionTests:
    def __init__(self):
        self.c = 3e8  # Speed of light
        self.test_results = {}
        
    def test_theoretical_range_resolution(self):
        """Test 1: Theoretical Range Resolution Validation"""
        print("\n" + "="*60)
        print("TEST 1: THEORETICAL RANGE RESOLUTION VALIDATION")
        print("="*60)
        
        # Test multiple bandwidths
        bandwidths = [50e6, 100e6, 200e6, 300e6]  # MHz
        results = []
        
        for B in bandwidths:
            # Create SAR model with this bandwidth
            sar = FinalSARModel(fc=10e9, B=B, Tp=10e-6)
            
            # Theoretical resolution
            theoretical_res = self.c / (2 * B)
            
            # Model calculated resolution
            calculated_res = sar.calculate_range_resolution()
            
            # Calculate error
            error_percent = abs(calculated_res - theoretical_res) / theoretical_res * 100
            
            results.append({
                'bandwidth_MHz': B/1e6,
                'theoretical_m': theoretical_res,
                'calculated_m': calculated_res,
                'error_percent': error_percent
            })
            
            print(f"Bandwidth: {B/1e6:6.1f} MHz | Theoretical: {theoretical_res:6.3f} m | "
                  f"Calculated: {calculated_res:6.3f} m | Error: {error_percent:8.6f}%")
        
        # Overall assessment
        max_error = max([r['error_percent'] for r in results])
        test_passed = max_error < 0.001  # Less than 0.001% error
        
        print(f"\nMaximum Error: {max_error:.8f}%")
        print(f"Test Result: {'PASS' if test_passed else 'FAIL'}")
        
        self.test_results['theoretical_validation'] = {
            'results': results,
            'max_error_percent': max_error,
            'passed': test_passed
        }
        
        return test_passed
    
    def test_two_point_target_separation(self):
        """Test 2: Two-Point Target Separation Test"""
        print("\n" + "="*60)
        print("TEST 2: TWO-POINT TARGET SEPARATION TEST")
        print("="*60)
        
        # Test parameters
        fc = 10e9  # 10 GHz
        B = 100e6  # 100 MHz
        Tp = 10e-6
        
        sar = FinalSARModel(fc=fc, B=B, Tp=Tp)
        theoretical_res = sar.calculate_range_resolution()
        
        print(f"Theoretical Range Resolution: {theoretical_res:.3f} m")
        
        # Test cases: targets separated by exactly the resolution limit
        test_cases = [
            {'R1': 1000, 'R2': 1000 + theoretical_res, 'description': 'At resolution limit'},
            {'R1': 1000, 'R2': 1000 + theoretical_res*0.8, 'description': 'Below resolution limit'},
            {'R1': 1000, 'R2': 1000 + theoretical_res*1.2, 'description': 'Above resolution limit'}
        ]
        
        results = []
        
        for i, case in enumerate(test_cases):
            print(f"\nCase {i+1}: {case['description']}")
            print(f"Target 1: {case['R1']} m, Target 2: {case['R2']} m")
            print(f"Separation: {case['R2'] - case['R1']:.3f} m")
            
            # Generate two-target response
            t1, response1 = sar.point_target_response(R0=case['R1'], plot=False)
            t2, response2 = sar.point_target_response(R0=case['R2'], plot=False)
            
            # Combine responses
            combined_response = response1 + response2
            
            # Range compress
            compressed = sar.range_compression(combined_response, plot=False)
            
            # Find peaks
            magnitude = np.abs(compressed)
            peaks_idx = []
            threshold = 0.5 * np.max(magnitude)  # 50% of max for peak detection
            
            # Simple peak detection
            for j in range(1, len(magnitude)-1):
                if (magnitude[j] > magnitude[j-1] and 
                    magnitude[j] > magnitude[j+1] and 
                    magnitude[j] > threshold):
                    peaks_idx.append(j)
            
            # Calculate separation between peaks if two found
            if len(peaks_idx) >= 2:
                # Convert to range
                range_axis = t1 * self.c / 2
                peak_ranges = [range_axis[idx] for idx in peaks_idx[:2]]
                measured_separation = abs(peak_ranges[1] - peak_ranges[0])
                
                # Calculate valley depth between peaks
                valley_idx = peaks_idx[0] + np.argmin(magnitude[peaks_idx[0]:peaks_idx[1]])
                valley_depth_db = 20 * np.log10(magnitude[valley_idx] / np.max(magnitude))
                
                resolved = valley_depth_db < -3  # 3dB criterion
                
                print(f"Peaks found: {len(peaks_idx)}")
                print(f"Measured separation: {measured_separation:.3f} m")
                print(f"Valley depth: {valley_depth_db:.1f} dB")
                print(f"Resolved (3dB criterion): {'YES' if resolved else 'NO'}")
                
                results.append({
                    'case': case['description'],
                    'theoretical_sep': case['R2'] - case['R1'],
                    'measured_sep': measured_separation,
                    'valley_depth_db': valley_depth_db,
                    'resolved': resolved,
                    'peaks_found': len(peaks_idx)
                })
            else:
                print(f"Peaks found: {len(peaks_idx)} (insufficient for separation test)")
                results.append({
                    'case': case['description'],
                    'theoretical_sep': case['R2'] - case['R1'],
                    'measured_sep': None,
                    'valley_depth_db': None,
                    'resolved': False,
                    'peaks_found': len(peaks_idx)
                })
        
        # Assessment
        expected_results = [True, False, True]  # At limit: maybe, Below: no, Above: yes
        test_passed = True
        for i, result in enumerate(results):
            if i == 0:  # At resolution limit - should be marginal
                continue
            elif i == 1:  # Below limit - should not resolve
                if result['resolved']:
                    test_passed = False
            elif i == 2:  # Above limit - should resolve
                if not result['resolved']:
                    test_passed = False
        
        print(f"\nOverall Test Result: {'PASS' if test_passed else 'FAIL'}")
        
        self.test_results['two_point_separation'] = {
            'results': results,
            'passed': test_passed
        }
        
        return test_passed
    
    def test_bandwidth_scaling(self):
        """Test 3: Range Resolution Bandwidth Scaling"""
        print("\n" + "="*60)
        print("TEST 3: BANDWIDTH SCALING TEST")
        print("="*60)
        
        bandwidths = np.array([25e6, 50e6, 100e6, 200e6, 400e6])
        expected_resolutions = self.c / (2 * bandwidths)
        measured_resolutions = []
        
        print("Bandwidth (MHz) | Expected Res (m) | Measured Res (m) | Error (%)")
        print("-" * 70)
        
        for B in bandwidths:
            sar = FinalSARModel(fc=10e9, B=B, Tp=10e-6)
            measured_res = sar.calculate_range_resolution()
            measured_resolutions.append(measured_res)
            
            expected_res = self.c / (2 * B)
            error = abs(measured_res - expected_res) / expected_res * 100
            
            print(f"{B/1e6:11.1f} | {expected_res:13.3f} | {measured_res:13.3f} | {error:8.4f}")
        
        measured_resolutions = np.array(measured_resolutions)
        
        # Check linear relationship (inverse of bandwidth)
        correlation = np.corrcoef(1/bandwidths, measured_resolutions)[0,1]
        
        # Check if all measurements are within 0.01% of expected
        max_error = np.max(np.abs(measured_resolutions - expected_resolutions) / expected_resolutions * 100)
        
        test_passed = correlation > 0.9999 and max_error < 0.01
        
        print(f"\nCorrelation with 1/B: {correlation:.8f}")
        print(f"Maximum Error: {max_error:.6f}%")
        print(f"Test Result: {'PASS' if test_passed else 'FAIL'}")
        
        self.test_results['bandwidth_scaling'] = {
            'bandwidths': bandwidths.tolist(),
            'expected_resolutions': expected_resolutions.tolist(),
            'measured_resolutions': measured_resolutions.tolist(),
            'correlation': correlation,
            'max_error_percent': max_error,
            'passed': test_passed
        }
        
        return test_passed
    
    def run_all_tests(self):
        """Run all range resolution tests"""
        print("RANGE RESOLUTION VALIDATION TEST SUITE")
        print("="*60)
        
        test_methods = [
            self.test_theoretical_range_resolution,
            self.test_two_point_target_separation,
            self.test_bandwidth_scaling
        ]
        
        passed_tests = 0
        total_tests = len(test_methods)
        
        for test_method in test_methods:
            try:
                if test_method():
                    passed_tests += 1
            except Exception as e:
                print(f"Test {test_method.__name__} failed with error: {e}")
        
        print("\n" + "="*60)
        print("RANGE RESOLUTION TEST SUMMARY")
        print("="*60)
        print(f"Tests Passed: {passed_tests}/{total_tests}")
        print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
        
        overall_passed = passed_tests == total_tests
        print(f"Overall Result: {'PASS' if overall_passed else 'FAIL'}")
        
        return overall_passed, self.test_results

if __name__ == "__main__":
    tester = RangeResolutionTests()
    success, results = tester.run_all_tests()
    
    # Save results for later analysis
    import json
    with open('../output/range_resolution_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: ../output/range_resolution_results.json")
