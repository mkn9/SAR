# Signal Processing Components

## Overview
Digital signal processing algorithms for radar applications including compression, filtering, and motion compensation.

## Components

### Range Compression (`range_compression.py`)
- Matched filtering
- Pulse compression
- Sidelobe suppression
- Windowing functions

### Azimuth Compression (`azimuth_compression.py`)
- Synthetic aperture processing
- Range cell migration correction
- Autofocus algorithms
- Image formation

### Motion Compensation (`motion_compensation.py`)
- INS/GPS integration
- Phase error correction
- Geometric distortion
- Registration algorithms

### Filtering (`filters.py`)
- FIR/IIR designs
- Adaptive filters
- Kalman filtering
- Spectral estimation

### Detection (`detection.py`)
- CFAR algorithms
- Threshold selection
- ROC analysis
- Track formation

## Usage Examples
```python
from radar.components.signal_processing import RangeCompression, MotionCompensation

# Setup processing chain
range_comp = RangeCompression(reference=chirp, window='hamming')
motion_comp = MotionCompensation(platform_data=gps_ins)

# Process SAR data
range_compressed = range_comp.process(raw_data)
motion_corrected = motion_comp.correct(range_compressed)
```
