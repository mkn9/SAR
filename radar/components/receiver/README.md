# Receiver Components

## Overview
RF receiver modules including LNA, mixers, ADC interfaces, and signal conditioning.

## Components

### Low Noise Amplifier (`lna.py`)
- Noise figure modeling
- Gain control
- Bandwidth optimization
- Temperature effects

### Mixer (`mixer.py`)
- Down-conversion
- LO leakage
- Image rejection
- IF filtering

### ADC Interface (`adc.py`)
- Sampling rate control
- Quantization effects
- Dynamic range
- Clock jitter

### Matched Filter (`matched_filter.py`)
- Correlation processing
- Range compression
- Sidelobe control
- Processing gain

## Usage Examples
```python
from radar.components.receiver import LNA, Mixer, MatchedFilter

# Create receiver chain
lna = LNA(gain=30, nf=1.5, frequency=10e9)
mixer = Mixer(lo_freq=9.9e9, if_freq=100e6)
mf = MatchedFilter(reference_signal=chirp)

# Process received signal
rx_signal = lna.amplify(input_signal)
if_signal = mixer.downconvert(rx_signal)
compressed = mf.process(if_signal)
```
