import numpy as np
import pandas as pd
import latentccm.datagen_utils as datagen
from latentccm.causal_inf import causal_score
from latentccm import DATADIR, EXPDIR

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import os

num_sims = 5
data_name = "Dpendulum_I"
embed_dim = 10
num_series = 3
time_lag = 400
#max_index_rec = 200000
#num_id_gp = 2000
sub_sample_rate_ccm = 10

for exp in range(num_sims):

    print(f"Start simulation {exp}")
    data_name_full = f"{data_name}_fold{exp}"


    metadata_dict = np.load(f"{DATADIR}/Dpendulum/data/{data_name}/fold_{exp}/{data_name_full}_joint_metadata.npy",allow_pickle = True).item()

    dt = metadata_dict["delta_t"]
    T = metadata_dict["T"]

    #Full trajectory causality

    y1 = np.load(f"{DATADIR}/Dpendulum/data/{data_name}/fold_{exp}/{data_name_full}_side0_full.npy")
    y2 = np.load(f"{DATADIR}/Dpendulum/data/{data_name}/fold_{exp}/{data_name_full}_side1_full.npy")
    if num_series==3:
        y3 = np.load(f"{DATADIR}/Dpendulum/data/{data_name}/fold_{exp}/{data_name_full}_side2_full.npy")
        x3 = y3[:,0]
    

    x1 = y1[:,0]
    x2 = y2[:,0]

    sc1_full, sc2_full = causal_score(x1,x2, lag = time_lag, embed = embed_dim, sub_sample_rate = sub_sample_rate_ccm)
    sc1_full_init, sc2_full_init = causal_score(x1,x2, lag = time_lag, embed = embed_dim, sub_sample_rate = sub_sample_rate_ccm, init = True)
    if num_series==3:
        sc13_full, sc31_full = causal_score(x1,x3,lag=time_lag,embed = embed_dim, sub_sample_rate = sub_sample_rate_ccm)
        sc23_full, sc32_full = causal_score(x2,x3,lag=time_lag,embed = embed_dim, sub_sample_rate = sub_sample_rate_ccm)
        sc13_full_init, sc31_full_init = causal_score(x1,x3,lag=time_lag,embed = embed_dim, sub_sample_rate = sub_sample_rate_ccm, init = True)
        sc23_full_init, sc32_full_init = causal_score(x2,x3,lag=time_lag,embed = embed_dim, sub_sample_rate = sub_sample_rate_ccm, init = True)

    else:
        sc13_full = 0
        sc31_full = 0
        sc23_full = 0
        sc32_full = 0

    score_full = sc1_full[-1]-sc2_full[-1]

    print(f"Done ! Score : {score_full}.")
    print(f"Score 1 : {sc1_full} - Score 2 : {sc2_full}")
    print(f"Score 13 : {sc13_full} - Score 31 : {sc31_full}")
    print(f"Score 23 : {sc23_full} - Score 32 : {sc32_full}")
    print(f"Score 1 INIT : {sc1_full_init} - Score 2 : {sc2_full_init}")
    print(f"Score 13 INIT : {sc13_full_init} - Score 31 : {sc31_full_init}")
    print(f"Score 23 INIT : {sc23_full_init} - Score 32 : {sc32_full_init}")

    print("Computing score on Linear reconstruction ...")

    # Linear reconstruction causality
    df = pd.read_csv(f"{DATADIR}/Dpendulum/data/{data_name}/fold_{exp}/{data_name_full}_joint_data.csv")

    #df = df.loc[df.ID<num_id_gp].copy()

    df.Time = df.Time + (df.ID-1)*10
    df.sort_values(by = "Time",inplace = True)
    new_cols = np.expand_dims(np.setdiff1d(np.linspace(0,df.Time.max(),int(df.Time.max()/dt)+1),df.Time.unique()),1)
    new_df = pd.DataFrame(np.concatenate((np.ones_like(new_cols),new_cols, np.zeros((new_cols.shape[0],(8*num_series)))),axis = 1), columns = df.columns)
    x3 = y3[0::10,0]
    df_ext = df.append(new_df)
    df_ext.sort_values(by="Time", inplace = True)
    df_ext.loc[df_ext.Mask_1==0,"Value_1"] = np.nan

    for value in [c for c in df_ext.columns if "Value" in c]:
        mask_name = "Mask_"+value.split("_")[-1]
        df_ext.loc[df_ext[mask_name]==0,value] = np.nan
        df_ext[value] = df_ext[value].interpolate(method="linear")

    #Remove  nans.
    df_ext.dropna(inplace = True)
    values_cols = [c for c in df_ext.columns if "Value" in c]
    y_interp = df_ext[values_cols].values

    sc1_lin, sc2_lin = causal_score(y_interp[:,0],y_interp[:,4], lag = time_lag, embed = embed_dim, sub_sample_rate = sub_sample_rate_ccm)
    sc1_lin_init, sc2_lin_init = causal_score(y_interp[:,0],y_interp[:,4], lag = time_lag, embed = embed_dim, sub_sample_rate = sub_sample_rate_ccm, init = True)
    
    if num_series==3:
        sc13_lin, sc31_lin = causal_score(y_interp[:,0],y_interp[:,8],lag=time_lag,embed = embed_dim, sub_sample_rate = sub_sample_rate_ccm)
        sc23_lin, sc32_lin = causal_score(y_interp[:,4],y_interp[:,8],lag=time_lag,embed = embed_dim, sub_sample_rate = sub_sample_rate_ccm)
        sc13_lin_init, sc31_lin_init = causal_score(y_interp[:,0],y_interp[:,8],lag=time_lag,embed = embed_dim, sub_sample_rate = sub_sample_rate_ccm, init = True)
        sc23_lin_init, sc32_lin_init = causal_score(y_interp[:,4],y_interp[:,8],lag=time_lag,embed = embed_dim, sub_sample_rate = sub_sample_rate_ccm, init = True)
    else:
        sc13_lin = 0
        sc31_lin = 0
        sc23_lin = 0
        sc32_lin = 0

    score_lin = sc1_lin[-1]-sc2_lin[-1]

    print(f"Done ! Score : {score_lin}.")
    print(f"Score 12 : {sc1_lin} - Score 21 : {sc2_lin}")
    print(f"Score 13 : {sc13_lin} - Score 31 : {sc31_lin}")
    print(f"Score 23 : {sc23_lin} - Score 32 : {sc32_lin}")
    print(f"Score 12 INIT : {sc1_lin_init} - Score 21 : {sc2_lin_init}")
    print(f"Score 13 INIT : {sc13_lin_init} - Score 31 : {sc31_lin_init}")
    print(f"Score 23 INIT : {sc23_lin_init} - Score 32 : {sc32_lin_init}")
    print("Computing GP interpolation ...")

    # GP processing
    
    df = pd.read_csv(f"{DATADIR}/Dpendulum/data/{data_name}/fold_{exp}/{data_name_full}_joint_data.csv")

    if num_series==3:
        dim_to_rec = [1,5,9]
    else:
        dim_to_rec = [1,5]

    kernel_dims = []
    for ix,val_idx in enumerate(dim_to_rec):
        df_comp_full = datagen.compress_df(df,fact=100).copy()
        
        df_ = df_comp_full.loc[df[f"Value_{val_idx}"]!=0].copy()
        df_ = df_.loc[df_.ID==2] #We learn on 100 samples.

        X = df_.Time.values.reshape(-1,1)
        y = df_[f"Value_{val_idx}"].values.reshape(-1,1)
        kernel = RBF(length_scale = 1)
        gpr = GaussianProcessRegressor(kernel = kernel, optimizer = 'fmin_l_bfgs_b', n_restarts_optimizer = 20).fit(X,y)

        print(gpr.kernel_)
        #y_ = gpr.predict((np.arange(index_y_start,index_y_end)*dt).reshape(-1,1))
        kernel_dims += [gpr.kernel_]

    print("Kernels Learnt")
        

    #Interpolation of the whole data
    y_gp = []

    for ix, val_idx in enumerate(dim_to_rec):
        df_ = df.loc[df[f"Value_{val_idx}"]!=0].copy()
        
        df_comp = datagen.compress_df(df_,fact=100).copy()
        t_comp  = 10*100 
        
        gp_pred = []
        for id_comp in df_comp.ID.unique():
            X = df_comp.loc[df_comp.ID==id_comp].Time.values.reshape(-1,1)
            y = df_comp.loc[df_comp.ID==id_comp][f"Value_{val_idx}"].values.reshape(-1,1)

            kernel = kernel_dims[ix]
            gpr = GaussianProcessRegressor(kernel = kernel, optimizer = None).fit(X,y)
            #print(gpr.kernel_)
            gp_pred += [gpr.predict(np.arange(0,t_comp,step = dt).reshape(-1,1))]
        
        y_gp += [np.concatenate(gp_pred)]

    print("Done. Computing causal inference ...")
    sc1_gp, sc2_gp = causal_score(y_gp[0][:,0],y_gp[1][:,0], lag = time_lag, embed = embed_dim, sub_sample_rate = sub_sample_rate_ccm)
    sc1_gp_init, sc2_gp_init = causal_score(y_gp[0][:,0],y_gp[1][:,0], lag = time_lag, embed = embed_dim, sub_sample_rate = sub_sample_rate_ccm, init= True)
    if num_series==3:
        sc13_gp, sc31_gp = causal_score(y_gp[0][:,0],y_gp[2][:,0],lag=time_lag,embed = embed_dim, sub_sample_rate = sub_sample_rate_ccm)
        sc23_gp, sc32_gp = causal_score(y_gp[1][:,0],y_gp[2][:,0],lag=time_lag,embed = embed_dim, sub_sample_rate = sub_sample_rate_ccm)
        sc13_gp_init, sc31_gp_init = causal_score(y_gp[0][:,0],y_gp[2][:,0],lag=time_lag,embed = embed_dim, sub_sample_rate = sub_sample_rate_ccm, init = True)
        sc23_gp_init, sc32_gp_init = causal_score(y_gp[1][:,0],y_gp[2][:,0],lag=time_lag,embed = embed_dim, sub_sample_rate = sub_sample_rate_ccm, init  = True)
    else:
        sc13_gp = 0
        sc31_gp = 0
        sc23_gp = 0
        sc32_gp = 0

    score_gp = sc1_gp[-1]-sc2_gp[-1] 
    
    print(f"Done! Score : {score_gp}")
    print(f"Score 1 : {sc1_gp} - Score 2 : {sc2_gp}")    
    print(f"Score 13 : {sc13_gp} - Score 31 : {sc31_gp}")
    print(f"Score 23 : {sc23_gp} - Score 32 : {sc32_gp}")
    print(f"Score 1 INIT : {sc1_gp_init} - Score 2 : {sc2_gp_init}")    
    print(f"Score 13 INIT: {sc13_gp_init} - Score 31 : {sc31_gp_init}")
    print(f"Score 23 INIT: {sc23_gp_init} - Score 32 : {sc32_gp_init}")

    print("Saving results ....")


    res_dict = metadata_dict.copy()
    res_dict.update({"score_full" : score_full,"score_linear" : score_lin,"score_gp" : score_gp , "dataset_name" : data_name })
    res_dict.update({"sc1_full": sc1_full[0], "sc2_full":sc2_full[0], "sc1_lin" : sc1_lin[0], "sc2_lin":sc2_lin[0], "sc1_gp": sc1_gp[0], "sc2_gp":sc2_gp[0]})
    res_dict.update({"sc13_full": sc13_full[0], "sc31_full":sc31_full[0], "sc23_full":sc23_full[0], "sc32_full":sc32_full[0]})
    res_dict.update({"sc13_lin": sc13_lin[0], "sc31_lin":sc31_lin[0], "sc23_lin":sc23_lin[0], "sc32_lin":sc32_lin[0]})
    res_dict.update({"sc13_gp": sc13_gp[0], "sc31_gp":sc31_gp[0], "sc23_gp":sc23_gp[0], "sc32_gp":sc32_gp[0]})

    outpath = f"{EXPDIR}/Dpendulum/results/results_ccm.csv"
    if os.path.exists(outpath):
        results_entry = pd.read_csv(outpath)

        results_update = results_entry.append(res_dict,ignore_index = True)
        results_update.to_csv(outpath,index = False)
    else:
        df = pd.DataFrame(res_dict)
        df.to_csv(outpath)










