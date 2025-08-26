#!/usr/bin/env python3
"""
SAR Benchmark Validation 2024 - Master Experiment Runner
Runs all SAR system validation experiments and generates comprehensive report
"""

import subprocess
import sys
import os
import datetime

def run_experiment(script_name, experiment_name):
    """Run a single experiment and capture results"""
    print(f"\n{'='*60}")
    print(f"RUNNING {experiment_name.upper()} EXPERIMENT")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print(f"✅ {experiment_name} experiment completed successfully")
            return True, result.stdout
        else:
            print(f"❌ {experiment_name} experiment failed")
            print(f"Error: {result.stderr}")
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {experiment_name} experiment timed out")
        return False, "Timeout"
    except Exception as e:
        print(f"💥 {experiment_name} experiment crashed: {e}")
        return False, str(e)

def generate_experiment_report(results):
    """Generate comprehensive experiment report"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
# SAR Benchmark Validation 2024 - Experiment Report
Generated: {timestamp}

## Experiment Overview
This experiment validates our SAR model implementation against three major 
operational SAR systems:
- TerraSAR-X (X-band, 9.65 GHz, 300 MHz bandwidth)
- RADARSAT-2 (C-band, 5.405 GHz, 100 MHz bandwidth)  
- Sentinel-1 (C-band, 5.405 GHz, 56.5 MHz bandwidth)

## Validation Methodology
Each SAR system is tested for:
1. Mathematical accuracy (range resolution, wavelength, chirp rate)
2. Point target response accuracy
3. Processing gain validation
4. Time-bandwidth product verification

## Results Summary
"""
    
    total_tests = len(results)
    passed_tests = sum(1 for success, _ in results.values() if success)
    
    report += f"- Total experiments: {total_tests}\n"
    report += f"- Successful experiments: {passed_tests}\n"
    report += f"- Success rate: {(passed_tests/total_tests)*100:.1f}%\n\n"
    
    for experiment_name, (success, output) in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        report += f"### {experiment_name}\n"
        report += f"Status: {status}\n\n"
        
        if success:
            # Extract key metrics from output
            lines = output.split('\n')
            for line in lines:
                if 'Status:' in line and ('✅' in line or '❌' in line):
                    report += f"- {line.strip()}\n"
        else:
            report += f"Error: {output[:200]}...\n"
        
        report += "\n"
    
    # Write report to file
    with open('../output/General/experiment_report.md', 'w') as f:
        f.write(report)
    
    print(f"\n📋 Comprehensive report saved to: ../output/General/experiment_report.md")
    return report

def main():
    """Run all SAR benchmark validation experiments"""
    print("🚀 Starting SAR Benchmark Validation 2024 Experiment Suite")
    print(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define experiments
    experiments = [
        ("terrasar_x_validation.py", "TerraSAR-X"),
        ("radarsat2_validation.py", "RADARSAT-2"),
        ("sentinel1_validation.py", "Sentinel-1")
    ]
    
    # Run all experiments
    results = {}
    
    for script, name in experiments:
        success, output = run_experiment(script, name)
        results[name] = (success, output)
    
    # Generate comprehensive report
    print(f"\n{'='*60}")
    print("GENERATING COMPREHENSIVE REPORT")
    print(f"{'='*60}")
    
    report = generate_experiment_report(results)
    
    # Final summary
    total_tests = len(results)
    passed_tests = sum(1 for success, _ in results.values() if success)
    success_rate = (passed_tests/total_tests)*100
    
    print(f"\n🎯 EXPERIMENT SUITE COMPLETE")
    print(f"{'='*60}")
    print(f"Total experiments: {total_tests}")
    print(f"Successful: {passed_tests}")
    print(f"Success rate: {success_rate:.1f}%")
    
    if success_rate >= 100:
        print("🎉 ALL EXPERIMENTS PASSED - EXCELLENT!")
    elif success_rate >= 75:
        print("✅ MOST EXPERIMENTS PASSED - GOOD!")
    else:
        print("⚠️  SOME EXPERIMENTS FAILED - NEEDS ATTENTION")
    
    print(f"\n📁 Results organized in:")
    print(f"  - ../output/TerraSAR-X/")
    print(f"  - ../output/RADARSAT-2/") 
    print(f"  - ../output/Sentinel-1/")
    print(f"  - ../output/General/")

if __name__ == "__main__":
    main()
