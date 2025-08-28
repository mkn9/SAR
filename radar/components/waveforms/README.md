# Waveform Components

## Overview
Radar waveform generation and analysis including chirps, coded waveforms, and OFDM signals.

## Components

### LFM Chirp (`lfm_chirp.py`)
- Linear frequency modulation
- Bandwidth control
- Time-bandwidth product
- Phase noise effects

### Phase Coded (`phase_coded.py`)
- Barker sequences
- Polyphase codes
- CAZAC sequences
- Correlation properties

### Stepped Frequency (`stepped_frequency.py`)
- Multi-frequency synthesis
- Range resolution
- Ambiguity analysis
- Coherent processing

### OFDM Waveforms (`ofdm.py`)
- Orthogonal subcarriers
- Cyclic prefix
- PAPR reduction
- Doppler resilience

### Noise Waveforms (`noise.py`)
- Pseudo-random sequences
- Correlation properties
- LPI characteristics
- Spectral shaping

## Usage Examples
```python
from radar.components.waveforms import LFMChirp, PhaseCoded

# Generate X-band LFM chirp
chirp = LFMChirp(fc=10e9, bandwidth=100e6, duration=10e-6)
signal = chirp.generate(sample_rate=200e6)

# Create Barker-13 coded pulse
barker = PhaseCoded(code_type='barker', length=13, chip_duration=1e-6)
coded_signal = barker.generate(carrier_freq=10e9)
```
