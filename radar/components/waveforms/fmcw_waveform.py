"""
FMCW Waveform Component
Based on Henrik Forstén's 6 GHz FMCW waveform design for quadcopter SAR

Specifications from Henrik Forstén's blog (hforsten.com):
- Waveform type: Linear Frequency Modulated Continuous Wave (LFMCW)
- Center frequency: 6 GHz
- Sweep bandwidth: 240 MHz
- Sweep time: 1 ms
- Chirp rate: 240 MHz/ms = 240 THz/s
- Modulation linearity: <1% deviation from linear
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from scipy import signal

@dataclass
class FMCWWaveformSpecs:
    """Henrik Forstén's FMCW Waveform Specifications"""
    
    # Core Waveform Parameters
    center_frequency: float = 6.0e9          # 6 GHz center frequency
    sweep_bandwidth: float = 240e6           # 240 MHz sweep bandwidth
    sweep_time: float = 1e-3                 # 1 ms sweep time
    waveform_type: str = "LFMCW"            # Linear FMCW
    
    # Frequency Planning
    start_frequency: float = 5.88e9          # 5.88 GHz start frequency
    stop_frequency: float = 6.12e9           # 6.12 GHz stop frequency
    chirp_rate: float = 240e12               # 240 THz/s chirp rate
    
    # Modulation Quality
    linearity_error_percent: float = 0.5     # 0.5% linearity error
    phase_noise_100khz: float = -90          # -90 dBc/Hz @ 100 kHz offset
    spurious_suppression: float = -60        # -60 dBc spurious suppression
    frequency_accuracy: float = 1e-6         # 1 ppm frequency accuracy
    
    # Timing Parameters
    ramp_up_time: float = 10e-6              # 10 μs ramp-up time
    ramp_down_time: float = 10e-6            # 10 μs ramp-down time
    flat_time_start: float = 5e-6            # 5 μs flat time at start
    flat_time_end: float = 5e-6              # 5 μs flat time at end
    
    # PLL Parameters (ADF4159)
    pll_reference_freq: float = 25e6         # 25 MHz reference frequency
    pll_step_size: float = 97.65625e3        # ~97.66 kHz step size
    pll_settling_time: float = 50e-6         # 50 μs PLL settling time
    pll_lock_time: float = 100e-6            # 100 μs PLL lock time
    
    # Waveform Synthesis
    synthesis_method: str = "DDS"            # Direct Digital Synthesis
    sample_rate: float = 1e6                 # 1 MSPS sample rate
    resolution_bits: int = 12                # 12-bit amplitude resolution
    
    # Range Performance
    range_resolution: float = 0.625          # 0.625 m range resolution
    max_unambiguous_range: float = 150.0     # 150 m max range
    range_sidelobes_db: float = -13.3        # -13.3 dB range sidelobes (rectangular)
    
    # Doppler Performance  
    max_doppler_freq: float = 2000           # 2 kHz max Doppler frequency
    doppler_resolution: float = 1000         # 1 kHz Doppler resolution
    doppler_ambiguity: float = 1000          # 1 kHz Doppler ambiguity
    
    # Power and Amplitude
    peak_power_dbm: float = 10.0             # 10 dBm peak power
    average_power_dbm: float = 10.0          # 10 dBm average power (CW)
    amplitude_stability: float = 0.1         # 0.1 dB amplitude stability
    
    # Environmental Stability
    temperature_drift_khz_c: float = 50      # 50 kHz/°C frequency drift
    aging_drift_ppm_year: float = 1.0        # 1 ppm/year aging drift
    vibration_sensitivity: float = 1e-9      # 1e-9 /g vibration sensitivity
    
    # Harmonic Content
    second_harmonic_db: float = -40          # -40 dB second harmonic
    third_harmonic_db: float = -45           # -45 dB third harmonic
    total_harmonic_distortion: float = -35   # -35 dB THD


class FMCWWaveform:
    """
    Henrik Forstén's FMCW Waveform Generator
    
    Based on his 6 GHz LFMCW design optimized for quadcopter SAR.
    Generates linear frequency modulated continuous wave signals.
    """
    
    def __init__(self, specs: Optional[FMCWWaveformSpecs] = None):
        self.specs = specs or FMCWWaveformSpecs()
        self.current_waveform = None
        self.phase_accumulator = 0.0
        
    def generate_ideal_chirp(self, sample_rate: float = 1e6, 
                           add_noise: bool = False) -> Dict:
        """
        Generate ideal FMCW chirp signal
        
        Args:
            sample_rate: Sample rate (Hz)
            add_noise: Add realistic phase noise and distortion
            
        Returns:
            Dictionary with waveform data
        """
        # Time vector
        t = np.arange(0, self.specs.sweep_time, 1/sample_rate)
        
        # Linear frequency sweep
        f_inst = (self.specs.start_frequency + 
                 self.specs.chirp_rate * t)
        
        # Phase calculation (integral of frequency)
        phase = 2 * np.pi * np.cumsum(f_inst) / sample_rate
        
        # Amplitude (constant for FMCW)
        amplitude = np.sqrt(10**(self.specs.peak_power_dbm/10) * 1e-3)
        
        # Generate complex signal
        signal_ideal = amplitude * np.exp(1j * phase)
        
        if add_noise:
            signal_ideal = self._add_realistic_impairments(signal_ideal, t)
            
        waveform_data = {
            'time': t,
            'signal': signal_ideal,
            'frequency': f_inst,
            'instantaneous_freq': np.gradient(phase) * sample_rate / (2 * np.pi),
            'amplitude': np.abs(signal_ideal),
            'phase': np.angle(signal_ideal),
            'sample_rate': sample_rate,
            'duration': self.specs.sweep_time
        }
        
        self.current_waveform = waveform_data
        return waveform_data
        
    def _add_realistic_impairments(self, signal: np.ndarray, time: np.ndarray) -> np.ndarray:
        """Add realistic impairments to ideal signal"""
        
        # 1. Phase noise (simplified 1/f model)
        phase_noise_std = 10**(self.specs.phase_noise_100khz/20) * 0.01  # Simplified
        phase_noise = np.cumsum(np.random.normal(0, phase_noise_std, len(signal)))
        
        # 2. Amplitude noise
        amp_noise_std = 10**(self.specs.amplitude_stability/20) * 0.01
        amp_noise = np.random.normal(1, amp_noise_std, len(signal))
        
        # 3. Frequency linearity error
        linearity_error = (self.specs.linearity_error_percent/100) * \
                         self.specs.sweep_bandwidth * np.sin(2*np.pi*time/self.specs.sweep_time)
        freq_error_phase = 2 * np.pi * np.cumsum(linearity_error) / len(signal)
        
        # 4. Spurious signals
        spurious_level = 10**(self.specs.spurious_suppression/20)
        spurious = spurious_level * np.sin(2*np.pi * 1.5e6 * time)  # 1.5 MHz spurious
        
        # Apply impairments
        impaired_signal = signal * amp_noise * np.exp(1j * (phase_noise + freq_error_phase))
        impaired_signal += spurious * np.exp(1j * np.random.uniform(0, 2*np.pi))
        
        return impaired_signal
        
    def generate_windowed_chirp(self, window_type: str = "hamming", 
                              sample_rate: float = 1e6) -> Dict:
        """
        Generate windowed FMCW chirp for sidelobe reduction
        
        Args:
            window_type: Window function type
            sample_rate: Sample rate (Hz)
            
        Returns:
            Dictionary with windowed waveform data
        """
        # Generate ideal chirp first
        chirp_data = self.generate_ideal_chirp(sample_rate, add_noise=True)
        
        # Generate window
        window_length = len(chirp_data['signal'])
        
        if window_type.lower() == "hamming":
            window = np.hamming(window_length)
        elif window_type.lower() == "hanning":
            window = np.hanning(window_length)
        elif window_type.lower() == "blackman":
            window = np.blackman(window_length)
        elif window_type.lower() == "kaiser":
            window = np.kaiser(window_length, beta=8.6)  # β=8.6 for ~60dB sidelobes
        else:
            window = np.ones(window_length)  # Rectangular (no window)
            
        # Apply window
        windowed_signal = chirp_data['signal'] * window
        
        # Calculate windowing loss
        windowing_loss_db = 20 * np.log10(np.mean(window))
        
        windowed_data = chirp_data.copy()
        windowed_data.update({
            'signal': windowed_signal,
            'window': window,
            'window_type': window_type,
            'windowing_loss_db': windowing_loss_db,
            'amplitude': np.abs(windowed_signal)
        })
        
        return windowed_data
        
    def calculate_range_profile(self, received_signal: np.ndarray, 
                              reference_chirp: np.ndarray) -> Dict:
        """
        Calculate range profile from received FMCW signal
        
        Args:
            received_signal: Received FMCW signal
            reference_chirp: Reference chirp for correlation
            
        Returns:
            Dictionary with range profile data
        """
        # Matched filtering (correlation)
        range_profile = np.correlate(received_signal, reference_chirp, mode='full')
        
        # Calculate range axis
        c = 3e8
        max_range = c * self.specs.sweep_time / (2 * self.specs.sweep_bandwidth) * len(range_profile)
        range_axis = np.linspace(0, max_range, len(range_profile))
        
        # Find valid range (positive delays only)
        center_idx = len(range_profile) // 2
        valid_range_profile = range_profile[center_idx:]
        valid_range_axis = range_axis[center_idx:] - range_axis[center_idx]
        
        # Calculate range resolution
        range_resolution = c / (2 * self.specs.sweep_bandwidth)
        
        return {
            'range_profile': valid_range_profile,
            'range_axis': valid_range_axis,
            'range_resolution': range_resolution,
            'peak_range': valid_range_axis[np.argmax(np.abs(valid_range_profile))],
            'dynamic_range_db': 20 * np.log10(np.max(np.abs(valid_range_profile)) / 
                                            np.mean(np.abs(valid_range_profile)))
        }
        
    def calculate_ambiguity_function(self, delay_range: Tuple[float, float] = (-1e-6, 1e-6),
                                   doppler_range: Tuple[float, float] = (-2000, 2000),
                                   resolution: Tuple[int, int] = (101, 101)) -> Dict:
        """
        Calculate radar ambiguity function for FMCW waveform
        
        Args:
            delay_range: Time delay range (s)
            doppler_range: Doppler frequency range (Hz)
            resolution: Resolution in delay and Doppler
            
        Returns:
            Dictionary with ambiguity function data
        """
        if self.current_waveform is None:
            self.generate_ideal_chirp()
            
        signal = self.current_waveform['signal']
        
        # Create delay and Doppler axes
        delays = np.linspace(delay_range[0], delay_range[1], resolution[0])
        dopplers = np.linspace(doppler_range[0], doppler_range[1], resolution[1])
        
        # Initialize ambiguity function
        ambiguity = np.zeros((len(delays), len(dopplers)), dtype=complex)
        
        # Calculate ambiguity function
        for i, delay in enumerate(delays):
            for j, doppler in enumerate(dopplers):
                # Time-shifted and Doppler-shifted signal
                delay_samples = int(delay * self.specs.sample_rate)
                
                if abs(delay_samples) < len(signal):
                    if delay_samples >= 0:
                        shifted_signal = np.concatenate([np.zeros(delay_samples), 
                                                       signal[:-delay_samples]])
                    else:
                        shifted_signal = np.concatenate([signal[-delay_samples:], 
                                                       np.zeros(-delay_samples)])
                    
                    # Apply Doppler shift
                    t = self.current_waveform['time']
                    doppler_signal = shifted_signal * np.exp(1j * 2 * np.pi * doppler * t)
                    
                    # Calculate correlation
                    correlation = np.sum(signal * np.conj(doppler_signal))
                    ambiguity[i, j] = correlation
                    
        # Normalize
        ambiguity = ambiguity / np.max(np.abs(ambiguity))
        
        return {
            'ambiguity_function': ambiguity,
            'delay_axis': delays,
            'doppler_axis': dopplers,
            'ambiguity_magnitude_db': 20 * np.log10(np.abs(ambiguity) + 1e-10),
            'range_axis': delays * 3e8 / 2,  # Convert delay to range
            'range_resolution_3db': self.specs.range_resolution,
            'doppler_resolution_3db': self.specs.doppler_resolution
        }
        
    def get_specifications(self) -> Dict:
        """Return complete waveform specifications"""
        return {
            "design_source": "Henrik Forstén SAR Blog (hforsten.com)",
            "application": "Quadcopter SAR FMCW Waveform",
            "waveform_type": self.specs.waveform_type,
            "center_frequency_ghz": self.specs.center_frequency / 1e9,
            "bandwidth_mhz": self.specs.sweep_bandwidth / 1e6,
            "sweep_time_ms": self.specs.sweep_time * 1000,
            "chirp_rate_thz_s": self.specs.chirp_rate / 1e12,
            "range_resolution_m": self.specs.range_resolution,
            "max_range_m": self.specs.max_unambiguous_range,
            "linearity_error_percent": self.specs.linearity_error_percent,
            "phase_noise_100khz_dbc": self.specs.phase_noise_100khz,
            "spurious_suppression_db": self.specs.spurious_suppression,
            "peak_power_dbm": self.specs.peak_power_dbm,
            "pll_chip": "ADF4159",
            "reference_frequency_mhz": self.specs.pll_reference_freq / 1e6,
            "drone_optimized": True,
            "continuous_wave": True
        }


# Component validation and test functions
def validate_waveform_specifications():
    """Validate Henrik Forstén's FMCW waveform specifications"""
    specs = FMCWWaveformSpecs()
    
    # Range resolution check
    c = 3e8
    calculated_range_res = c / (2 * specs.sweep_bandwidth)
    assert abs(calculated_range_res - specs.range_resolution) < 0.01
    
    # Chirp rate check
    calculated_chirp_rate = specs.sweep_bandwidth / specs.sweep_time
    assert abs(calculated_chirp_rate - specs.chirp_rate) < 1e9
    
    # Frequency range check
    freq_span = specs.stop_frequency - specs.start_frequency
    assert abs(freq_span - specs.sweep_bandwidth) < 1e6
    
    # Center frequency check
    calculated_center = (specs.start_frequency + specs.stop_frequency) / 2
    assert abs(calculated_center - specs.center_frequency) < 1e6
    
    print("✅ Henrik Forstén FMCW Waveform specifications validated")


if __name__ == "__main__":
    # Demonstrate Henrik Forstén's FMCW waveform
    waveform = FMCWWaveform()
    
    print("Henrik Forstén's 6 GHz FMCW Waveform for Quadcopter SAR")
    print("=" * 60)
    
    specs = waveform.get_specifications()
    for key, value in specs.items():
        print(f"{key}: {value}")
    
    # Generate waveform
    print("\nGenerating FMCW Waveform:")
    chirp_data = waveform.generate_ideal_chirp(sample_rate=2e6, add_noise=True)
    print(f"Samples generated: {len(chirp_data['signal'])}")
    print(f"Duration: {chirp_data['duration']*1000:.1f} ms")
    print(f"Frequency range: {chirp_data['frequency'][0]/1e9:.3f} - {chirp_data['frequency'][-1]/1e9:.3f} GHz")
    
    # Generate windowed version
    windowed_data = waveform.generate_windowed_chirp("hamming", sample_rate=2e6)
    print(f"Windowing loss: {windowed_data['windowing_loss_db']:.2f} dB")
    
    # Calculate ambiguity function
    print("\nCalculating Ambiguity Function:")
    ambiguity_data = waveform.calculate_ambiguity_function()
    print(f"Ambiguity function size: {ambiguity_data['ambiguity_function'].shape}")
    print(f"Range resolution (3dB): {ambiguity_data['range_resolution_3db']:.3f} m")
    
    validate_waveform_specifications()
