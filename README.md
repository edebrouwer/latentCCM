
# Latent Convergent Cross Mapping

Implementation of the Latent CCM paper (https://openreview.net/forum?id=4TSiOTkKe5P)

![latent_CCM_new (1)](https://user-images.githubusercontent.com/22655671/116911247-e1bed400-ac46-11eb-9164-e869f6a11c87.png)

## Installation

`poetry install`

## Latent CCM in a nutshell

We provide a simple example of using latent CCM for inference of causal link direction in time series in the notebook latentCCM.ipynb.

`/latentccm/latentCCM.ipynb` for the worked out example on Lorenz dynamical systems.

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
