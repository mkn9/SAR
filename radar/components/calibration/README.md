# Calibration Components

## Overview
Radar system calibration routines for phase/amplitude correction, temperature compensation, and performance validation.

## Components

### Internal Calibration (`internal_cal.py`)
- Built-in test signals
- Loop-back paths
- Coupler networks
- Real-time correction

### External Calibration (`external_cal.py`)
- Corner reflector targets
- Active transponders
- Distributed targets
- Absolute calibration

### Phase Correction (`phase_correction.py`)
- Phase drift compensation
- Temperature effects
- Aging compensation
- Multi-channel alignment

### Amplitude Correction (`amplitude_correction.py`)
- Gain variations
- Frequency response
- Dynamic range
- Linearity correction

### Temperature Compensation (`temp_compensation.py`)
- Thermal drift models
- Sensor integration
- Predictive correction
- Adaptive algorithms

## Usage Examples
```python
from radar.components.calibration import InternalCal, PhaseCorrection

# Setup calibration system
cal_system = InternalCal(cal_tone_freq=10.1e9, coupling=-20)
phase_cal = PhaseCorrection(reference_channel=0)

# Perform calibration
cal_data = cal_system.measure_response()
corrections = phase_cal.calculate_corrections(cal_data)
corrected_data = phase_cal.apply_corrections(raw_data, corrections)
```
