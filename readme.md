# EncodingNeuronalShape
Code for the article *Encoding neuronal shape in the stochastic dynamics of branching processes.*
Tools for neuronal tree analysis and simulation, organized as two Python packages:

- `DendroTree`: graph/tree utilities and morphology metrics
- `SimulationTree`: Simulation code for the growth of 2D sensory neurons

## Requirements

- Linux or WSL on Windows is recommended
- Python 3.10+ 
- `ffmpeg` for MP4 animation export: <https://ffmpeg.org/download.html>

## Setup

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

## Run Simulations

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
