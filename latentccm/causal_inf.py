

from skccm.utilities import train_test_split


#Embed the time series
import skccm as ccm
import numpy as np

def causal_score(x1,x2, lag = 40, embed = 8,full_path = False, sub_sample_rate = 1.0,init = False):
    #prop delay embeds is the proportion of delay embedings to use for the CCM.
    lag = lag
    embed = embed
    e1 = ccm.Embed(x1)
    e2 = ccm.Embed(x2)
    X1 = e1.embed_vectors_1d(lag,embed)
    X2 = e2.embed_vectors_1d(lag,embed)

    X1 = X1[0::sub_sample_rate,:]
    X2 = X2[0::sub_sample_rate,:]

    #split the embedded time series
    x1tr, x1te, x2tr, x2te = train_test_split(X1,X2, percent=.75)

    CCM = ccm.CCM() #initiate the class

    #library lengths to test
    len_tr = len(x1tr)
    if full_path:
        lib_lens = np.arange(100, int(len_tr), int(len_tr)/20, dtype='int')
    else:
        if init:
            lib_lens = [int(1000)]
        else:
            lib_lens = [int(len_tr)]

    #test causation
    CCM.fit(x1tr,x2tr)
    x1p, x2p = CCM.predict(x1te, x2te,lib_lengths=lib_lens)

    sc1,sc2 = CCM.score()

    return sc1, sc2

def causal_score_direct(X1,X2,init = False):
    
    #split the embedded time series
    x1tr, x1te, x2tr, x2te = train_test_split(X1,X2, percent=.75)

    CCM = ccm.CCM() #initiate the class

    #library lengths to test
    len_tr = len(x1tr)
    #lib_lens = np.arange(10, len_tr, len_tr/20, dtype='int')
    if init:
        lib_lens = [100]
    else:
        lib_lens = [int(len_tr)]

    #test causation
    CCM.fit(x1tr,x2tr)
    x1p, x2p = CCM.predict(x1te, x2te,lib_lengths=lib_lens)

    sc1,sc2 = CCM.score()

    return sc1, sc2

def embed_time_series(x,lag,embed_dim):
    num_x = x.shape[0]-(embed_dim-1)*lag
    embed_list = []
    for i in range(embed_dim):
        embed_list.append(x[(embed_dim-1)*lag-(i*lag):(embed_dim-1)*lag-(i*lag)+num_x].reshape(-1,1))
    return np.concatenate(embed_list,axis=-1)

def CCM_compute(X1,X2):
    #split the embedded time series
    x1tr, x1te, x2tr, x2te = train_test_split(X1,X2, percent=.75)

    CCM = ccm.CCM() #initiate the class

    #library lengths to test
    len_tr = len(x1tr)
    lib_lens = np.arange(100, len_tr, len_tr/10, dtype='int')

    #test causation
    CCM.fit(x1tr,x2tr)
    x1p, x2p = CCM.predict(x1te, x2te,lib_lengths=lib_lens)

    sc1,sc2 = CCM.score()
    return sc1, sc2
