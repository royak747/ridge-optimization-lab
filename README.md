# Exploring Optimization Strategies for Strongly Convex Ridge Regression under Noise and Conditioning

This project enables systematic evaluation of gradient-based optimization methods under varying conditioning and noise levels, with both synthetic and real-world datasets.

## Layout

```text
config.py
src/
  config.py
  data_generation.py
  standard_gradient_descent.py
  gradient_descent_ridge.py
  sgd_ridge.py
  UCI_data_gen.py
  get_convergence_factor.py
scripts/
  run_data_generation.py
  run_standard_gd.py
  run_ridge_gd.py
  run_gd_tradeoffs.py
  run_UCI_experiment.py
  run_convergence_rates_experiment.py
artifacts/            
data/   
reports/               
```

## Setup

For MacOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

#### Synthetic Data Generation

First, generate ill-conditioned datasets with varying levels of conditioning and noise:

```bash
python -m scripts.run_data_generation
```

This generates 12 datasets across combinations of condition numbers (κ) and noise levels. These datasets are saved as a pkl file in `data` for later reference.


#### Optimization Experiments

Run the following scripts to evaluate optimization performance:

```bash
python -m scripts.run_standard_gd
python -m scripts.run_ridge_gd
python -m scripts.run_gd_tradeoffs
python -m scripts.run_convergence_rates_experiment
```

#### UCI Dataset Experiment

```bash
python -m scripts.run_UCI_experiment
```

## Outputs
* `data`: location of synthetic datasets stored in a `.pkl` file which are used for optimization experiment
* `artifacts`: location of generated graphs after running optimization experiments