# Microwave Structure AI Design

Neural network models for forward and inverse design of microwave coupled-line structures. Predicts S-parameters from design parameters and vice versa, with dual-frequency optimization capabilities.

## Features
- *Forward Prediction*: S11/S21 from design parameters + frequency
- *Inverse Design*: Design parameters from target S-parameters
- *Dual-Band Optimization*: Simultaneous dual-frequency design
- *Hybrid Approach*: Combines TensorFlow & PyTorch models

## Models
- DANN.py: TensorFlow-based forward/inverse models
- Hybridpipeline.py: Hybrid TensorFlow+PyTorch with CVAE

## Quick Start
```bash
python DANN.py  # Run TensorFlow model
python Hybridpipeline.py  # Run hybrid model
