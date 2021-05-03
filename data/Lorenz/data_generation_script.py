import numpy as np
import pandas as pd
from latentccm import datagen_utils as datagen
from latentccm.causal_inf import causal_score
from latentccm import DATADIR
import os



if __name__ =="__main__":

    num_sims = 5
    experiment_name_base = "Lorenz_I"

    for exp in range(num_sims):

        exp_dir = os.path.join(DATADIR,"Lorenz","data",experiment_name_base,f"fold_{exp}")
        os.makedirs(exp_dir, exist_ok = True)
        
        print(f"Start simulation {exp}")
        experiment_name = f"{experiment_name_base}_fold{exp}"
        data_name = experiment_name

        df, y, metadata_dict = datagen.generate_Lorenz_data(seed = exp)

        
        np.save(f"{exp_dir}/{data_name}_side0_metadata.npy",metadata_dict)
            
        np.save(f"{exp_dir}/{data_name}_side1_metadata.npy",metadata_dict)

        np.save(f"{exp_dir}/{data_name}_side0_full.npy",y[:,:3])
        np.save(f"{exp_dir}/{data_name}_side1_full.npy",y[:,3:])
    

