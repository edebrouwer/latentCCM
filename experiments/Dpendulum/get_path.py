import torch
import tqdm
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os
import pandas as pd
from latentccm import data_utils
from latentccm import datagen_utils as datagen
from latentccm import DATADIR, EXPDIR
from latentccm.fitting_models import models

#import matplotlib
#matplotlib.use("agg")
#import matplotlib.pyplot as plt



def get_both_paths(model_name_base, outfile_base, data_name, fold, variant_name, return_hidden, num_series,noise_std, random_shift):

    for i in range(num_series):
        model_name = model_name_base + f"_side{i}"
        print(f"Reconstructing time series from {model_name}")
        dataset = f"{DATADIR}/Dpendulum/data/{data_name}/fold_{fold}/{data_name}_fold{fold}_side{i}_data.csv"
        outfile = outfile_base+f"_side{i}.csv"
         
        if "shuffled" in outfile_base:
            num_aggregated_series = 0
        else:
            num_aggregated_series = 100

        df_recs, random_lag = get_path(model_name,variant_name, dataset, return_hidden, num_aggregated_series = num_aggregated_series,noise_std = noise_std, random_shift = random_shift)
        df_recs[0].to_csv(outfile, index = False)
        if random_lag is not None:
            np.save(outfile[:-4]+"_random_lag.npy", random_lag)
            print(f"Saved random_lag in {outfile[:-4]}_random_lag.npy")
    return 0



def get_path_index(model_name, variant_name, dataset, return_hidden = False, indexes = [0,1,2,3], T = 10, val_prop = 0.8, num_collate = 1, home_path =  None, noise_std = 0):
    device = torch.device("cuda")

    params_dict = np.load(f"{home_path}trained_models/{model_name}_params.npy", allow_pickle = True).item()

    metadata = params_dict['metadata']
    params_dict = params_dict['model_params']

    N       = metadata["num_series"]
    delta_t = params_dict["delta_t"]


    #collate_fn=lambda x : data_utils.discrete_collate_fn(x, delta_t = False)   
    df1 = pd.read_csv(dataset)
    
    if noise_std>0:
        df1 = data_utils.add_noise_to_df(df1, noise_std) 
    
    df1 = datagen.compress_df(df1,num_collate,T//num_collate)
    df1 = df1.loc[df1.ID.isin(indexes)]
    df1.ID = df1.ID.map(dict(zip(indexes,np.arange(len(indexes)))))

    #T = df1.Time.max()+0.1
    #T = 10
    
    val_options = {"T_val":int(val_prop*T),"max_val_samples":10}
    data_val_1   = data_utils.ODE_Dataset(panda_df = df1,validation = True,val_options = val_options)

    dl_val_1 = DataLoader(dataset=data_val_1, collate_fn=data_utils.custom_collate_fn, shuffle=False, batch_size = 100,num_workers=1)
    #dl_val_2 = DataLoader(dataset=data_val_2, collate_fn=data_utils.custom_collate_fn, shuffle=True, batch_size = 1,num_workers=1)


    model = models.NNFOwithBayesianJumps(input_size = params_dict["input_size"], hidden_size = params_dict["hidden_size"],
                                            p_hidden = params_dict["p_hidden"], prep_hidden = params_dict["prep_hidden"],
                                            logvar = params_dict["logvar"], mixing = params_dict["mixing"],
                                            full_gru_ode = params_dict["full_gru_ode"],
                                            solver = params_dict["solver"], impute = params_dict["impute"],store_hist = True)


    model.to(device)

    model.load_state_dict(torch.load(f"{home_path}trained_models/{model_name}{variant_name}.pt"))

    dl_val_list = [dl_val_1]

    dfs_rec = []
    for dl_val in dl_val_list:
        with torch.no_grad():
            for i, b in enumerate(dl_val):
                times    = b["times"]
                time_ptr = b["time_ptr"]
                X        = b["X"].to(device)
                M        = b["M"].to(device)
                obs_idx  = b["obs_idx"]
                cov      = b["cov"].to(device)

                y = b["y"]

                hT, loss, _, t_vec, p_vec, h_vec, eval_times, eval_vals = model(times, time_ptr, X, M, obs_idx, delta_t=delta_t, T=T, cov=cov, return_path=True)

                if params_dict["solver"] =="euler":
                    eval_vals = p_vec
                    eval_times = t_vec
                else:
                    eval_times = eval_times.cpu().numpy()

                mu, v = torch.chunk(eval_vals[:,:,:],2, dim = 2)
                mu = mu.cpu().numpy()
                v = v.cpu().numpy()

                observations = X.cpu().numpy()
               
                if params_dict["logvar"]:
                    up   = mu + np.exp(0.5*v) * 1.96
                    down = mu - np.exp(0.5*v) * 1.96
                else:
                    up   = mu + np.sqrt(v) * 1.96
                    down = mu - np.sqrt(v) * 1.96

                break

        if return_hidden:
            mu = h_vec.cpu().numpy()

        round_time = np.expand_dims(np.round(eval_times,3),1)
        columns = ["ID","Time"] + [f"Value_{i}" for i in range(1,mu.shape[2]+1)]

        df_rec = []
        for sim_num in range(mu.shape[1]):
            
            y_to_fill = np.concatenate((sim_num*np.ones_like(round_time),round_time,mu[:,sim_num,:]),1)
            df_rec_ = pd.DataFrame(y_to_fill, columns = columns)
            df_rec_.drop_duplicates(subset = ["Time"], keep = "last", inplace = True)
            df_rec += [df_rec_]
       
        df_rec = pd.concat(df_rec)

        dfs_rec += [df_rec]

        #plt.figure()
        #for c in columns[1:]:
        #    plt.plot(df_rec.Time.values,df_rec[c].values)
        #    plt.scatter(df1.Time.values,df1[c].values)
        #plt.savefig("reconstruction_try.pdf")
        
    return dfs_rec


def get_path(model_name, variant_name, dataset, return_hidden = False, num_aggregated_series = 100, noise_std = 0, random_shift = 0):
    device = torch.device("cuda")


    params_dict = np.load(f"{EXPDIR}/Dpendulum/trained_models/{model_name}_params.npy", allow_pickle = True).item()

    metadata = params_dict['metadata']
    params_dict = params_dict['model_params']

    N       = metadata["num_series"]
    delta_t = metadata["delta_t"]

    #c_12 = c_12
    #c_21 = c_21


    #dfs, y = datagen.Coupled_Double_pendulum_sample(T = T, dt = delta_t, l1 = metadata["l1"], l2 = metadata["l2"], m1 = metadata["m1"], m2 = metadata["m2"], c_12 = c_12, c_21 = c_21, noise_level = metadata["noise_level"], sample_rate = metadata["sample_rate"], multiple_sample_rate = metadata["multiple_sample_rate"] )


    #df1,_ = datagen.scaling(dfs[0],y[0])
    #df2,_ = datagen.scaling(dfs[1],y[1])

    #df1.ID = 0
    #df2.ID = 0
    #collate_fn=lambda x : data_utils.discrete_collate_fn(x, delta_t = False)   
    df1 = pd.read_csv(dataset)
    original_N = df1.ID.nunique()
    
    if num_aggregated_series==0:
        num_aggregated_series = df1.ID.nunique()

    df1 = datagen.compress_df(df1,original_N/num_aggregated_series,10)
  
    if random_shift >0:
        random_lag = np.round(2*np.random.random(df1.ID.nunique()),2)
        df1["Time"] = df1["Time"] + np.repeat(random_lag,df1.groupby("ID").size().values)

    if noise_std>0:
        df1 = data_utils.add_noise_to_df(df1, noise_std) 

    T = df1.Time.max()+0.1
    T = 10 * (original_N/num_aggregated_series) + random_shift

    data_val_1   = data_utils.ODE_Dataset(panda_df = df1)


    dl_val_1 = DataLoader(dataset=data_val_1, collate_fn=data_utils.custom_collate_fn, shuffle=False, batch_size = num_aggregated_series,num_workers=1)
    #dl_val_2 = DataLoader(dataset=data_val_2, collate_fn=data_utils.custom_collate_fn, shuffle=True, batch_size = 1,num_workers=1)


    if "discrete" not in model_name:
        model = models.NNFOwithBayesianJumps(input_size = params_dict["input_size"], hidden_size = params_dict["hidden_size"],
                                            p_hidden = params_dict["p_hidden"], prep_hidden = params_dict["prep_hidden"],
                                            logvar = params_dict["logvar"], mixing = params_dict["mixing"],
                                            full_gru_ode = params_dict["full_gru_ode"],
                                            solver = params_dict["solver"], impute = params_dict["impute"],store_hist = True)

    else:
        model = models.Discretized_GRU(input_size = params_dict["input_size"], hidden_size = params_dict["hidden_size"],
                                        p_hidden = params_dict["p_hidden"], prep_hidden = params_dict["prep_hidden"],
                                        logvar = params_dict["logvar"], mixing = params_dict["mixing"],
                                        dropout_rate = params_dict["dropout"],
                                     impute = params_dict["impute"])




    model.to(device)

    model.load_state_dict(torch.load(f"{EXPDIR}/Dpendulum/trained_models/{model_name}{variant_name}.pt"))

    dl_val_list = [dl_val_1]

    dfs_rec = []
    for dl_val in dl_val_list:
        with torch.no_grad():
            for i, b in enumerate(dl_val):
                times    = b["times"]
                time_ptr = b["time_ptr"]
                X        = b["X"].to(device)
                M        = b["M"].to(device)
                obs_idx  = b["obs_idx"]
                cov      = b["cov"].to(device)

                y = b["y"]

                if "discrete" in model_name:
                    h_init = 0.1*torch.randn(cov.shape[0],params_dict["hidden_size"],device =device)
                    hT, loss, mse, _, t_vec, p_vec, h_vec = model(times, time_ptr, X, M, obs_idx, delta_t=delta_t,h_init = h_init, T=T, cov=cov, return_path=True)
                else:
                    hT, loss, _, t_vec, p_vec, h_vec, eval_times, eval_vals = model(times, time_ptr, X, M, obs_idx, delta_t=delta_t, T=T, cov=cov, return_path=True)

                if params_dict["solver"] =="euler":
                    eval_vals = p_vec
                    eval_times = t_vec
                else:
                    eval_times = eval_times.cpu().numpy()

                mu, v = torch.chunk(eval_vals[:,:,:],2, dim = 2)
                mu = mu.cpu().numpy()
                v = v.cpu().numpy()

                observations = X.cpu().numpy()
               
                if params_dict["logvar"]:
                    up   = mu + np.exp(0.5*v) * 1.96
                    down = mu - np.exp(0.5*v) * 1.96
                else:
                    up   = mu + np.sqrt(v) * 1.96
                    down = mu - np.sqrt(v) * 1.96
                

                #plt.figure()
                #colors=["orange","green","red","blue"]
                #dims = [0,1,2,3]
                #plt.plot(eval_times,mu[:,0,dims[0]],"-.", c= colors[0])
                #plt.plot(eval_times,mu[:,0,dims[1]],"-.", c= colors[1])
                #plt.plot(eval_times,mu[:,0,dims[2]],"-.", c= colors[2])
                #plt.plot(eval_times,mu[:,0,dims[3]],"-.", c= colors[3])
                #for dim in range(4):
                #    observed_idx = np.where(M.cpu().numpy()[:,dims[dim]]==1)[0]
                #    plt.scatter(times[observed_idx],observations[observed_idx,dims[dim]], c = colors[dim])
                
                #plt.scatter(eval_times,-1.2*np.ones(len(eval_times)),marker="|", label = f"DOPRI : {len(eval_times)} evals", c="green")
                #plt.legend(loc = 7)
                #plt.title("Prediction of trajectories for Double pendulum")
                #plt.savefig("DPendulum_debug.pdf")
                #plt.close()

                break

        if return_hidden:
            mu = h_vec.cpu().numpy()

        round_time = np.expand_dims(np.round(eval_times,3),1)
        columns = ["ID","Time"] + [f"Value_{i}" for i in range(1,mu.shape[2]+1)]

        df_rec = []
        for sim_num in range(mu.shape[1]):
            
            y_to_fill = np.concatenate((sim_num*np.ones_like(round_time),round_time,mu[:,sim_num,:]),1)
            df_rec_ = pd.DataFrame(y_to_fill, columns = columns)
            df_rec_.drop_duplicates(subset = ["Time"], keep = "last", inplace = True)
            df_rec += [df_rec_]
       
        df_rec = pd.concat(df_rec)

        dfs_rec += [df_rec]

        #plt.figure()
        #for c in columns[1:]:
        #    plt.plot(df_rec.Time.values,df_rec[c].values)
        #    plt.scatter(df1.Time.values,df1[c].values)
        #plt.savefig("reconstruction_try.pdf")
    
    if random_shift>0:
        return dfs_rec, random_lag
    else:
        return dfs_rec, None

if __name__ =="__main__":
    #df_recs = get_path(c_12 = 0, c_21 = 0)
    #df_recs[0].to_csv(outfile, index = False)
    
    folds = [0]
    data_name = "Dpendulum_I"
    out_variants = ["_shuffled_hidden","_shuffled","_hidden",""]

    noise_std = 0
    num_series = 3
    random_shift = 0
    
    for fold in folds:
        for out_variant in out_variants:
            
            os.makedirs(os.path.join(EXPDIR,"Dpendulum","reconstructions"),exist_ok = True)

            outfile_base = f"{EXPDIR}/Dpendulum/reconstructions/{data_name}{out_variant}_fold{fold}"
            model_name_base = f"{data_name}_fold{fold}"
            
            data_name_full = f"{data_name}_fold{fold}"
            variant_name = f"_BEST_VAL_MSE"
    
            return_hidden = "hidden" in outfile_base
            if "noisy" not in outfile_base:
                noise_std = 0

            get_both_paths(model_name_base = model_name_base,
            outfile_base = outfile_base,
            data_name = data_name,
            fold = fold,
            variant_name = variant_name, return_hidden = return_hidden, num_series = num_series,noise_std = noise_std, random_shift = random_shift)
    
            
