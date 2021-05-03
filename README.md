# Latent Convergent Cross Mapping

Implementation of the Latent CCM paper (https://openreview.net/forum?id=4TSiOTkKe5P)

## Installation

`poetry install`

## Experiments

### Data Generation

`cd data/Dpendulum`

`python data_generation_scipt.py`

### Training the ODE models on the Dependulum (Irregular) data 

`cd experiments/Dpendulum`

`./launch_gru_ode.sh`

The models are then saved in `trained_models` folder.
 
### Reconstruction of the trajectories

`python get_path.py`

### Causal direction computation (CCM)

`python gruode_scores.py`

The results are then saved in the `results` folder.
