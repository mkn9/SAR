# Antenna Components

## Overview
Antenna modeling, pattern calculations, and beamforming algorithms for radar systems.

## Components

### Array Factor (`array_factor.py`)
- Linear/planar array patterns
- Element spacing optimization
- Grating lobe analysis
- Beam steering

### Beamforming (`beamforming.py`)
- Digital beamforming
- Adaptive algorithms
- Null steering
- MVDR/LCMV methods

### Pattern Analysis (`pattern_analysis.py`)
- Gain calculations
- Sidelobe levels
- Beamwidth measurement
- Polarization effects

### Pointing Control (`pointing.py`)
- Mechanical steering
- Electronic steering
- Tracking algorithms
- Calibration methods

## Usage Examples
```python
from radar.components.antenna import ArrayFactor, Beamforming

# Design 64-element linear array
array = ArrayFactor(elements=64, spacing=0.5, frequency=10e9)
beamformer = Beamforming(array_geometry=array.geometry)

# Calculate pattern and steer beam
pattern = array.calculate_pattern(theta_range=(-90, 90))
steered_weights = beamformer.steer_beam(azimuth=30, elevation=0)
```
