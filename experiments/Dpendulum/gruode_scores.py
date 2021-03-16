import os
import numpy as np
import pandas as pd

from latentccm.causal_inf import causal_score, causal_score_direct, embed_time_series
from latentccm import DATADIR, EXPDIR

samples_per_sec = 100
time_bins = 10 #seconds
embed_dim = 1
time_lag = 200
num_time_series = 3
time_prop_h = 0.4
subsample_rate = 10
prop_delay_embed = 1

original_time_series = False

data_name = "Dpendulum_I"
folds = [0]

for fold in folds:
    print(f"Computing fold : {fold} ...")
    data_name_full = f"{data_name}_fold{fold}"
    reconstruction_name = f"{data_name}_shuffled_hidden_fold{fold}"

    df_ode_list = []
    for series in range(num_time_series):

        df_o = pd.read_csv(f"{DATADIR}/Dpendulum/data/{data_name}/fold_{fold}/{data_name_full}_side{series}_data.csv")
        df_r = pd.read_csv(f"{EXPDIR}/Dpendulum/reconstructions/{reconstruction_name}_side{series}.csv")
        y = np.load(f"{DATADIR}/Dpendulum/data/{data_name}/fold_{fold}/{data_name_full}_side{series}_full.npy")

        if "random_shift" in reconstruction_name:
            random_lag = np.load(f"{EXPDIR}/Dependulum/reconstructions/{reconstruction_name}_side{series}_random_lag.npy")
            df_r.Time = df_r.Time - np.repeat(random_lag,df_r.groupby("ID").size().values)
            df_r.drop(df_r.loc[df_r.Time<0].index,inplace = True)
            df_r.drop(df_r.loc[df_r.Time>10].index, inplace = True)
    

        n_chunks = df_o.ID.nunique()/100

        df_r.Time = df_r.Time + df_r.ID*(n_chunks*samples_per_sec)
        df_ode0 = df_r.copy()
        
        if original_time_series:

            df_full = pd.DataFrame(y,columns = [f"Value_{i+1}" for i in range(4)])
            df_full["ID"] = np.repeat(np.arange(1000),1000)[:-1] 
            df_ode_list += [df_full]
        else:
            df_ode_list += [df_ode0]

    if "hidden" in reconstruction_name:
        if "shuffle" in reconstruction_name:
            x_list = []
            for df_ode in df_ode_list:
                embed_list = []
                val_c = [c for c in df_ode.columns if "Value" in c]
                for series_index in range(int(df_ode.ID.max())):
                    df_ = df_ode.loc[df_ode.ID==series_index,val_c]
                    limit_t = int(df_.shape[0]*(1-time_prop_h))
                    if embed_dim>1: #delay embedding the hiddens.
                        embed_list.append(embed_time_series(df_.values[limit_t:], time_lag, embed_dim)[0::subsample_rate])
                    else:
                        embed_list.append(df_.values[limit_t:][0::subsample_rate])
                x_list.append(np.concatenate(embed_list,axis=0))
        else:
            x_list = [df_ode[[c for c in df_ode.columns if "Value" in c]].values[0::subsample_rate] for df_ode in df_ode_list]
    else:
        if "shuffle" in reconstruction_name:
            x_list = []
            for df_ode in df_ode_list:
                embed_list = []
                for series_index in range(int(df_ode.ID.max())):
                    df_ = df_ode.loc[df_ode.ID==series_index]
                    limit_t = int(df_.shape[0]*(1-time_prop_h))
                    embed_list.append(embed_time_series(df_.Value_1.values[limit_t:],time_lag,embed_dim))
                x_list.append(np.concatenate(embed_list[0::subsample_rate],axis=0))
        else:
            x_list = [df_ode.Value_1.values for df_ode in df_ode_list]



    if ("hidden" in reconstruction_name) or ("shuffle" in reconstruction_name):
        print("computing match between hiddens ...")
        sc1_gruode, sc2_gruode = causal_score_direct(x_list[0],x_list[1])
        sc1_gruode_init, sc2_gruode_init = causal_score_direct(x_list[0],x_list[1],init = True)

        if num_time_series==3:
            sc13_gruode, sc31_gruode = causal_score_direct(x_list[0],x_list[2])
            sc23_gruode, sc32_gruode = causal_score_direct(x_list[1],x_list[2])
            
            sc13_gruode_init, sc31_gruode_init = causal_score_direct(x_list[0],x_list[2],init = True)
            sc23_gruode_init, sc32_gruode_init = causal_score_direct(x_list[1],x_list[2],init = True)



    else:    
        sc1_gruode, sc2_gruode = causal_score(x_list[0],x_list[1], lag = time_lag, embed = embed_dim, sub_sample_rate = subsample_rate)
        
        if num_time_series==3:
            sc13_gruode, sc31_gruode = causal_score(x_list[0],x_list[2],lag= time_lag,embed = embed_dim, sub_sample_rate = subsample_rate)
            sc23_gruode, sc32_gruode = causal_score(x_list[1],x_list[2],lag= time_lag,embed = embed_dim, sub_sample_rate = subsample_rate)


    print(f"sc1 : {sc1_gruode} - sc2 {sc2_gruode}")
    print(f"sc1 init : {sc1_gruode_init} - sc2 init : {sc2_gruode_init}")
    if num_time_series==3:
        print(f"sc31 : {sc31_gruode} - sc13 {sc13_gruode}")
        print(f"sc31 init : {sc31_gruode_init} - sc13 init : {sc13_gruode_init}")
        print(f"sc32 : {sc32_gruode} - sc23 {sc23_gruode}")
        print(f"sc32 init : {sc32_gruode_init} - sc23 init : {sc23_gruode_init}")
        

    results_path = f"{EXPDIR}/Dpendulum/results/results_ccm.csv"
    if os.path.exists(results_path):
        results_entry = pd.read_csv(results_path)

        results_entry.loc[results_entry.dataset_name==data_name,"sc1_gru_ode"] = sc1_gruode
        results_entry.loc[results_entry.dataset_name==data_name,"sc2_gru_ode"] = sc2_gruode
        results_entry.loc[results_entry.dataset_name==data_name,"sc1_gru_ode_init"] = sc1_gruode_init
        results_entry.loc[results_entry.dataset_name==data_name,"sc2_gru_ode_init"] = sc2_gruode_init
        if num_time_series==3:
            results_entry.loc[results_entry.dataset_name==data_name,"sc13_gru_ode"] = sc13_gruode
            results_entry.loc[results_entry.dataset_name==data_name,"sc31_gru_ode"] = sc31_gruode
            results_entry.loc[results_entry.dataset_name==data_name,"sc23_gru_ode"] = sc23_gruode
            results_entry.loc[results_entry.dataset_name==data_name,"sc32_gru_ode"] = sc32_gruode




        results_entry.to_csv(results_path,index = False)



def compress_df(df,fact,time_bin = 10):
    df["IDbis"] = (df.ID-1) % fact
    df.Time = df.Time + time_bin * df.IDbis
    df.ID = df.ID - df.IDbis
    df.ID = df.ID.map(dict(zip(df.ID.unique(),np.arange(df.ID.nunique()))))
    df.drop("IDbis", axis = 1,inplace = True)
    return df
