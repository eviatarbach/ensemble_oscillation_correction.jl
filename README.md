# Code for "Ensemble Oscillation Correction (EnOC): Leveraging oscillatory modes to improve forecasts of chaotic systems"

This repository contains the code for the paper "[Ensemble Oscillation Correction (EnOC): Leveraging oscillatory modes to improve forecasts of chaotic systems](https://doi.org/10.1175/JCLI-D-20-0624.1)" by Eviatar Bach, Safa Mote, V. Krishnamurthy, A. Surjalal Sharma, Michael Ghil, and Eugenia Kalnay.

All the code was written by Eviatar Bach. You can contact me with any questions at eviatarbach@protonmail.com.

## Dependencies

Julia:
- [Distributions.jl](https://github.com/JuliaStats/Distributions.jl)
- [NearestNeighbors.jl](https://github.com/KristofferC/NearestNeighbors.jl)

Python:
- [parasweep](https://github.com/eviatarbach/parasweep) (optional, to run parameter sweeps)
- [xarray](https://github.com/pydata/xarray)
- [xskillscore](https://github.com/xarray-contrib/xskillscore)

## Description of files

- **analog.jl**: Functions for analog forecasting and mapping to the oscillation subspace.
- **da.jl**: An ensemble transform Kalman filter.
- **integrators.jl**: The Runge--Kutta 4th-order integrator.
- **models.jl**: The tendency functions for each toy model.
- **ssa.jl**: Functions for decomposing and reconstructing a signal using multi-channel singular spectrum analysis (M-SSA).
- **ssa_varimax.jl**: Utility functions for varimax SSA.
