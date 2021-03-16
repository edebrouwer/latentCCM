import numpy as np
import pandas as pd
from latentccm import datagen_utils as datagen 
from latentccm.causal_inf import causal_score
from latentccm import DATADIR

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

import os

c_21 = 0 #Pendulum 2 drives pendulum 1. 2 -> 1
c_12 = 0
c_13 = 0
c_31 = 0.5
c_32 = 1

num_pendulums = 3

sample_rate = 4

T = 10000

noise_level = 0.01

l_p1 = [0.5,1]
l_p2 = [0.5,1]
l_p3 = [1,1]
m_p1 = [2., 1.]
m_p2 = [2., 1.]
m_p3 = [1.,3.]

theta1 = 1
theta2 = -0.5

lps = [l_p1, l_p2, l_p3]
mps = [m_p1, m_p2, m_p3]


dt = 0.01
sample_rate = sample_rate
multiple_sample_rate = 0.3
num_sims = 5
experiment_name_base = "Dpendulum_I"


for exp in range(num_sims):

    exp_dir = os.path.join(DATADIR,"Dpendulum","data",experiment_name_base,f"fold_{exp}")
    os.makedirs(exp_dir, exist_ok = True)

    print(f"Start simulation {exp}")
    experiment_name = f"{experiment_name_base}_fold{exp}"
    data_name = experiment_name

    sampling_not_at_random = True

    #Data generation
    
    seed = 421 + exp


    dfs, ys = datagen.Coupled_Double_pendulum_sample(T = T, dt = dt, c_12 = c_12, c_21 = c_21, c_31 = c_31, c_32 = c_32, l_p1 = l_p1, l_p2 = l_p2, l_p3 = l_p3, m_p1 = m_p1, m_p2 = m_p2, m_p3 = m_p3,noise_level = noise_level, sample_rate = sample_rate, multiple_sample_rate = multiple_sample_rate,seed = seed, sampling_not_at_random = sampling_not_at_random)

    # SAVE BOTH Datasets.
    df_list = []
    for i in range(num_pendulums):
        df1,y_s1 = datagen.scaling(dfs[i], ys[i])
        df_ = df1.copy()
        
        df_.sort_values(by = "Time",inplace = True)
        bins = np.linspace(0, T, int(T/10)+1)
        id_vec = np.expand_dims(np.digitize(df_.Time.values,bins),1)
        df_["Time"] = df_.Time.values - bins[id_vec-1][:,0]
        df_["ID"] = id_vec
        
        index_end = 8
        df_1 = df_.loc[(df_.Value_1!=0)&(df_.ID==index_end)]
        df__1 = df1.loc[(df1.Value_1!=0) & (df1.Time>((index_end-1)*10)) &(df1.Time<((index_end)*10))]
        
        df_.to_csv(f"{exp_dir}/{data_name}_side{i}_data.csv",index= False)
        np.save(f"{exp_dir}/{data_name}_side{i}_full.npy",y_s1)
        
        #Save metadata dictionary
        lp = lps[i]
        mp = mps[i]
        metadata_dict = {"T":T, "delta_t":dt, "l1": lp[0], "l2": lp[1],
                    "m1" : mp[0], "m2": mp[1], "noise_level" : noise_level,
                    "num_series" : T/10,"c_21": c_21, "c_12":c_12, "c_31":c_31, "c_32": c_32,
                    "sample_rate": sample_rate, "multiple_sample_rate": multiple_sample_rate,"seed" :seed}
        np.save(f"{exp_dir}/{data_name}_side{i}_metadata.npy",metadata_dict)
        
        df_list += [df_]
        
    #joint dataset   
    name_dict = {"Value_1":"Value_5","Value_2":"Value_6","Value_3":"Value_7","Value_4":"Value_8",
                "Mask_1":"Mask_5","Mask_2":"Mask_6","Mask_3":"Mask_7","Mask_4":"Mask_8"}

    name_dict2 = {"Value_1":"Value_9","Value_2":"Value_10","Value_3":"Value_11","Value_4":"Value_12",
                "Mask_1":"Mask_9","Mask_2":"Mask_10","Mask_3":"Mask_11","Mask_4":"Mask_12"}

    df_m = df_list[0].merge(df_list[1].rename(columns = name_dict),on = ["ID","Time"], how = "outer")
    if num_pendulums==3:
        df_m = df_m.merge(df_list[2].rename(columns = name_dict2), on = ["ID","Time"], how = "outer")
    df_m.sort_values(by=["ID","Time"],inplace = True)
    df_m.fillna(0.0, inplace = True)

    df_m.to_csv(f"{exp_dir}/{data_name}_joint_data.csv",index = False)
    np.save(f"{exp_dir}/{data_name}_joint_full.npy",y_s1)

    metadata_dict = {"T":T, "delta_t":dt, "l_p1": l_p1, "l_p2": l_p2,
                    "m_p1" : m_p1, "m_p2": m_p2, "noise_level" : noise_level,
                    "num_series" : T/10, "c_21": c_21, "c_12":c_12, "c_31": c_31, "c_32": c_32,
                    "sample_rate": sample_rate, "multiple_sample_rate": multiple_sample_rate,"seed" :seed}
    np.save(f"{exp_dir}/{data_name}_joint_metadata.npy",metadata_dict)


    print("Generated Data.")

