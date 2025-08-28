# Transmitter Components

## Overview
RF transmitter modules for radar systems including power amplifiers, modulators, and signal generation.

## Components

### Power Amplifier (`power_amplifier.py`)
- Linear and nonlinear PA models
- Efficiency calculations
- Thermal management
- Distortion analysis

### Modulator (`modulator.py`)
- I/Q modulation
- Phase/frequency modulation
- Digital predistortion
- Carrier generation

### Chirp Generator (`chirp_generator.py`)
- LFM waveform synthesis
- Bandwidth control
- Phase noise modeling
- Timing accuracy

## Usage Examples
```python
from radar.components.transmitter import PowerAmplifier, ChirpGenerator

# Create 1kW X-band transmitter
pa = PowerAmplifier(power=1000, frequency=10e9, efficiency=0.65)
chirp = ChirpGenerator(fc=10e9, B=100e6, duration=10e-6)

# Generate transmit signal
tx_signal = pa.amplify(chirp.generate())
```
