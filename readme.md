# EncodingNeuronalShape
Code for the article *Encoding neuronal shape in the stochastic dynamics of branching processes.*
Tools for neuronal tree analysis and simulation, organized as two Python packages:

- `DendroTree`: graph/tree utilities and morphology metrics
- `SimulationTree`: Simulation code for the growth of 2D sensory neurons

`DataAnalysis` contains the data and code used for statistical tests.

## Table of contents

- [Installation](#installation)
- [Run Simulations of dendritic morphogenesis](#run-simulations-of-dendritic-morphogenesis)
- [DataAnalysis](#dataanalysis)

## Installation

### Requirements

- Linux or WSL on Windows is recommended
- Python 3.10+ 
- `ffmpeg` for MP4 animation export: <https://ffmpeg.org/download.html>

### Setup

Clone the repository, then create and activate an environment:

```bash
conda create -n neuronal_shape python=3.12
conda activate neuronal_shape
```

Go to the repository and install in this order:

```bash
pip install ./DendroTree
pip install ./SimulationTree
pip install ./DataAnalysis
```

## Run Simulations of dendritic morphogenesis
This step is optional for running the DataAnalysis part which contains cached version of the data in the "raw_data" subfolders.

Go to the simulation package directory:

```bash
cd SimulationTree
```

Run one of the provided scripts:

- Three models from the article:

```bash
MPLBACKEND=Agg python run_simulations/script_launch_simu_3_models.py
```

- Parameter variation for anomalous diffusion (`lambda`, `kappa`, `sigma`, `alpha`):

```bash
MPLBACKEND=Agg python run_simulations/script_launch_parameters_variation.py
```

- Exploration of front-speed scaling vs. anomalous diffusion parameters:

```bash
MPLBACKEND=Agg python run_simulations/script_launch_simu_exploration.py
```

Outputs are written to directories whose names start with `simulation_result`.

## DataAnalysis

This part contains 9 scripts:
### 1. Morphometrics analysis
Plot morphometrics of class I and class IV neurons.
Results: [DataAnalysis/Result/1_morphometrics](DataAnalysis/Result/1_morphometrics)

### 2. Branching rate analysis
Estimate branching rate trends and compare class I, class IV, and class IV MT-inhibited conditions.
Results: [DataAnalysis/Result/2_branching_rate](DataAnalysis/Result/2_branching_rate)

### 3. Post-contact analysis
Measure post-contact branch behavior (memory time and displacement) and compare classes.
Results: [DataAnalysis/Result/3_post_contact](DataAnalysis/Result/3_post_contact)

### 4. Front speed and switching analysis
Analyze growth/retraction switching, drift, and diffusion from high-frame-rate trajectories, as well as the dependence between phase speed and duration.
Results: [DataAnalysis/Result/4_von_voff_ton_toff_slope_duration](DataAnalysis/Result/4_von_voff_ton_toff_slope_duration)

### 5. Increment statistics (`alpha`, `sigma`, `mu`)
Estimate increment-model parameters per class and test class differences.
Results: [DataAnalysis/Result/5_increments_alpha_sigma_mu](DataAnalysis/Result/5_increments_alpha_sigma_mu)

### 6. Simulation morphometrics
Compute morphometrics on simulated trees and plot comparison with experimental data for each of the three models of the article.
Results: [DataAnalysis/Result/6_simulation_morphometrics](DataAnalysis/Result/6_simulation_morphometrics)

### 7. Parameter variation analysis
Quantify how morphometrics on simulated trees change when model parameters are varied.
Results: [DataAnalysis/Result/7_parameter_variation](DataAnalysis/Result/7_parameter_variation)

### 8. Tubulin MT-only invasion
Measure microtubule invasion speed in class I vs class IV.
Results: [DataAnalysis/Result/8_tubulin_mt_only_invasion](DataAnalysis/Result/8_tubulin_mt_only_invasion)

### 9. SI front propagation
Generate supplementary front-propagation figures.
Results: [DataAnalysis/Result/9_SI_front_propagation](DataAnalysis/Result/9_SI_front_propagation)
