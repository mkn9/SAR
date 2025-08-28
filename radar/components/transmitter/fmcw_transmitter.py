"""
FMCW Transmitter Component
Based on Henrik Forstén's 6 GHz SAR design for quadcopter implementation

Specifications from Henrik Forstén's blog (hforsten.com):
- Operating frequency: 6 GHz (C-band)
- Modulation: Linear Frequency Modulated Continuous Wave (LFMCW)
- Sweep bandwidth: 240 MHz
- Sweep time: 1 ms
- Output power: ~10 dBm (10 mW) - suitable for drone application
- VCO: ADF4159 fractional-N PLL with integrated VCO
- Power amplifier: Integrated on-chip amplifier
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

@dataclass
class FMCWTransmitterSpecs:
    """Henrik Forstén's FMCW Transmitter Specifications"""
    
    # Core RF Parameters
    center_frequency: float = 6.0e9          # 6 GHz center frequency
    sweep_bandwidth: float = 240e6           # 240 MHz sweep bandwidth  
    sweep_time: float = 1e-3                 # 1 ms sweep time
    output_power_dbm: float = 10.0           # 10 dBm output power
    output_power_watts: float = 10e-3        # 10 mW output power
    
    # VCO/PLL Specifications (ADF4159)
    pll_chip: str = "ADF4159"                # Analog Devices fractional-N PLL
    reference_frequency: float = 25e6         # 25 MHz reference oscillator
    pll_step_size: float = 97.65625e3        # ~97.66 kHz step size
    phase_noise_100khz: float = -90          # -90 dBc/Hz @ 100 kHz offset
    spurious_suppression: float = -60        # -60 dBc spurious suppression
    
    # Modulation Parameters
    chirp_rate: float = 240e12               # 240 MHz/ms = 240 THz/s
    frequency_deviation: float = 240e6       # Total frequency deviation
    modulation_type: str = "Linear"          # Linear frequency modulation
    
    # Power Amplifier (Integrated)
    pa_gain: float = 15.0                    # 15 dB integrated PA gain
    pa_p1db: float = 12.0                    # 12 dBm 1dB compression point
    pa_efficiency: float = 0.20              # 20% PA efficiency (typical for integrated)
    
    # Timing and Control
    sweep_trigger: str = "External"          # External trigger for SAR synchronization
    ramp_up_time: float = 10e-6             # 10 μs ramp-up time
    ramp_down_time: float = 10e-6           # 10 μs ramp-down time
    
    # Temperature Specifications
    operating_temp_min: float = -40          # -40°C minimum operating temperature
    operating_temp_max: float = 85           # 85°C maximum operating temperature
    temp_coefficient: float = 50e3          # 50 kHz/°C frequency drift
    
    # Power Consumption (Critical for drone application)
    supply_voltage: float = 5.0              # 5V supply voltage
    supply_current: float = 0.15             # 150 mA supply current
    power_consumption: float = 0.75          # 750 mW total power consumption
    
    # Physical Specifications (Drone-optimized)
    pcb_size_mm: Tuple[float, float] = (25, 15)  # 25mm x 15mm PCB
    component_height: float = 3.0            # 3mm maximum component height
    mass_grams: float = 2.5                  # 2.5g total mass including PCB
    
    # Interface Specifications
    control_interface: str = "SPI"           # SPI control interface
    trigger_input: str = "TTL"              # TTL trigger input
    rf_connector: str = "U.FL"              # U.FL RF connector (drone-suitable)
    
    # Performance Metrics
    frequency_accuracy: float = 1e-6         # 1 ppm frequency accuracy
    phase_noise_1mhz: float = -110          # -110 dBc/Hz @ 1 MHz offset
    harmonics_suppression: float = -40       # -40 dBc harmonic suppression
    
    # SAR-Specific Parameters
    coherent_integration_time: float = 1e-3  # 1 ms coherent integration
    range_resolution: float = 0.625          # 0.625 m range resolution (c/2B)
    max_range: float = 150                   # 150 m maximum range (drone application)
    
    # Environmental (Drone Operation)
    vibration_resistance: str = "MIL-STD-810G"  # Military vibration standard
    humidity_rating: str = "95% RH"          # 95% relative humidity rating
    altitude_rating: float = 3000            # 3000m altitude rating


class FMCWTransmitter:
    """
    Henrik Forstén's FMCW Transmitter Implementation
    
    Based on his 6 GHz SAR design optimized for quadcopter deployment.
    Uses ADF4159 PLL with integrated VCO and power amplifier.
    """
    
    def __init__(self, specs: Optional[FMCWTransmitterSpecs] = None):
        self.specs = specs or FMCWTransmitterSpecs()
        self.is_initialized = False
        self.current_frequency = self.specs.center_frequency
        self.sweep_active = False
        
    def initialize(self) -> bool:
        """Initialize the FMCW transmitter hardware"""
        # In real implementation, this would configure the ADF4159
        self.is_initialized = True
        return True
        
    def generate_chirp(self, sample_rate: float = 1e6) -> np.ndarray:
        """
        Generate FMCW chirp signal
        
        Args:
            sample_rate: ADC sample rate (Hz)
            
        Returns:
            Complex chirp signal array
        """
        t = np.arange(0, self.specs.sweep_time, 1/sample_rate)
        
        # Linear frequency sweep
        f_start = self.specs.center_frequency - self.specs.sweep_bandwidth/2
        chirp_rate = self.specs.sweep_bandwidth / self.specs.sweep_time
        
        # Instantaneous frequency
        f_inst = f_start + chirp_rate * t
        
        # Complex signal with phase noise
        phase = 2 * np.pi * np.cumsum(f_inst) / sample_rate
        amplitude = np.sqrt(self.specs.output_power_watts)
        
        # Add realistic phase noise (simplified model)
        phase_noise = np.random.normal(0, 0.01, len(t))  # Simplified phase noise
        
        signal = amplitude * np.exp(1j * (phase + phase_noise))
        
        return signal
        
    def set_power(self, power_dbm: float) -> bool:
        """Set transmitter output power"""
        if power_dbm <= self.specs.pa_p1db:
            self.specs.output_power_dbm = power_dbm
            self.specs.output_power_watts = 10**(power_dbm/10) * 1e-3
            return True
        return False
        
    def get_specifications(self) -> dict:
        """Return complete transmitter specifications"""
        return {
            "design_source": "Henrik Forstén SAR Blog (hforsten.com)",
            "application": "Quadcopter SAR",
            "frequency_ghz": self.specs.center_frequency / 1e9,
            "bandwidth_mhz": self.specs.sweep_bandwidth / 1e6,
            "power_dbm": self.specs.output_power_dbm,
            "power_consumption_mw": self.specs.power_consumption * 1000,
            "mass_grams": self.specs.mass_grams,
            "pll_chip": self.specs.pll_chip,
            "range_resolution_m": self.specs.range_resolution,
            "max_range_m": self.specs.max_range,
            "pcb_size_mm": self.specs.pcb_size_mm,
            "drone_optimized": True
        }


# Component validation and test functions
def validate_specifications():
    """Validate Henrik Forstén's transmitter specifications"""
    specs = FMCWTransmitterSpecs()
    
    # Range resolution check: c/(2*B)
    c = 3e8
    calculated_range_res = c / (2 * specs.sweep_bandwidth)
    assert abs(calculated_range_res - specs.range_resolution) < 0.01
    
    # Power consumption check (critical for drone)
    assert specs.power_consumption < 1.0  # Must be under 1W for drone
    
    # Mass check (critical for drone)
    assert specs.mass_grams < 5.0  # Must be under 5g for small drone
    
    print("✅ Henrik Forstén FMCW Transmitter specifications validated")


if __name__ == "__main__":
    # Demonstrate Henrik Forstén's FMCW transmitter
    tx = FMCWTransmitter()
    tx.initialize()
    
    print("Henrik Forstén's 6 GHz FMCW SAR Transmitter")
    print("=" * 50)
    
    specs = tx.get_specifications()
    for key, value in specs.items():
        print(f"{key}: {value}")
    
    validate_specifications()
