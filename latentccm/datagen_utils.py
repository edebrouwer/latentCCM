import numpy as np
import pandas as pd
import argparse
#from .dpendulum import DPendulum 
#from dynsys import DampedDrivenPendulum

#import matplotlib
#matplotlib.use('agg')
#import matplotlib.pyplot as plt
from numpy import cos, sin

class DPendulum:
    #http://scienceworld.wolfram.com/physics/DoublePendulum.html
    def __init__(self, l1=1.0, l2=1.0, m1=1.0, m2=1.0, theta1 = -1., theta2 = 0.5):
        #Set params
        self.coupled = []
        self.l1 = l1
        self.l2 = l2
        self.m1 = m1
        self.m2 = m2
        self.g = 9.81
        #Set initial conditions
        self.theta1 = theta1 + 0.05*np.random.randn()
        self.theta2 = theta2 + 0.05*np.random.randn()
        self.p1     = 0
        self.p2     = 0
    def __C1(self):
        tmp  = self.p1 * self.p2 * sin(self.theta1 - self.theta2)
        tmp /= self.l1 * self.l2  * (self.m1 + self.m2 * sin(self.theta1 - self.theta2)**2)
        return tmp
    def __C2(self):
        tmp  = self.l2**2 * self.m2 * self.p1**2 + self.l1**2 * (self.m1 + self.m2) * self.p2**2 
        tmp -= self.l1 * self.l2 * self.m2 * self.p1 * self.p2 * cos(self.theta1 - self.theta2)
        tmp /= 2* (self.l1 * self.l2 * (self.m1 + self.m2 * sin(self.theta1 - self.theta2)**2))**2
        tmp *= sin(2*(self.theta1 - self.theta2))
        return tmp
    def dtheta1(self):
        tmp  = self.l2 * self.p1 - self.l1 * self.p2 * cos(self.theta1 - self.theta2) 
        tmp /= self.l1**2 * self.l2 * (self.m1 + self.m2 * sin(self.theta1 - self.theta2)**2)
        return tmp
    def dtheta2(self):
        tmp  = self.l1*(self.m1 + self.m2)* self.p2-self.l2 * self.m2 * self.p1 * cos (self.theta1 - self.theta2)
        tmp /= self.l1 * self.l2**2 * self.m2 * (self.m1 + self.m2 * sin(self.theta1 - self.theta2)**2)
        return tmp
    def dp1(self):
        coupl_term = 0
        if len(self.coupled) > 0:
            for (w, dpend) in self.coupled:
                coupl_term += w * 2 * (self.theta1 - dpend.theta1)
        return -(self.m1 + self.m2) * self.g * self.l1 * sin(self.theta1) - self.__C1() + self.__C2() - coupl_term
    def dp2(self):
        return - self.m2 * self.g * self.l2 * sin(self.theta2) + self.__C1() - self.__C2()
    
    def leapfrog_step(self,tau):
        dp1 = self.dp1()
        dp2 = self.dp2()
        self.p1 += dp1 * tau / 2
        self.p2 += dp2 * tau / 2
        
        dtheta1 = self.dtheta1()
        dtheta2 = self.dtheta2()
        self.theta1 += dtheta1 * tau
        self.theta2 += dtheta2 * tau
        
        dp1 = self.dp1()
        dp2 = self.dp2()
        self.p1 += dp1 * tau / 2
        self.p2 += dp2 * tau / 2

    def couple(self, dpend, w):
        self.coupled += [(w, dpend)]


def df_dt(x ,dt, sigma, rho, beta):
    dx = sigma * (x[1]-x[0]) * dt
    dy = (x[0]*(rho-x[2])-x[1])*dt
    dz = (x[0]*x[1] - beta*x[2])*dt
    return np.array([dx,dy,dz])

def Lorenz(T, dt, sigma= 10, rho= 28, beta=8/3, noise_level= 0.01, couplings = [0,0]):
    N_t  = int(T//dt)
    
    x_l1 = np.zeros((N_t,3))
    x_l1[0,:] = np.array([10,15,21.1])

    x_l2 = np.zeros((N_t,3))
    x_l2[0,:] = np.array([17,12,14.2])

    x_l3 = np.zeros((N_t,3))
    x_l3[0,:] = np.array([3,8,12.4])


    for i in range(1,N_t):
        x_l1[i,:] = x_l1[i-1] + df_dt(x_l1[i-1],dt, sigma, rho, beta) + couplings[1] * np.array([x_l1[i-1,1]-x_l2[i-1,0],0,0]) * dt + noise_level*np.random.randn(3)
        x_l2[i,:] = x_l2[i-1] + df_dt(x_l2[i-1],dt, sigma, rho, beta) + couplings[0] * np.array([x_l2[i-1,1]-x_l1[i-1,0],0,0]) * dt + noise_level*np.random.randn(3)
        x_l3[i,:] = x_l3[i-1] + df_dt(x_l3[i-1],dt, sigma, rho, beta)

    return x_l1, x_l2, x_l3

def Lorenz_sample(T,dt,sigma,rho,beta, noise_level, couplings, sample_rate, multiple_sample_rate, num_series = 1, seed=432):
    '''
    Samples from the Lorenz time series
    The sample rate should be expressed in samples per unit of time. (on average there will be sample_rate*T sample per series)
    The dual_sample rate gives the proportion of samples wich are jointly sampled (for both dimensions)
    We generate dummy covariates (all 0)
    '''
    np.random.seed(seed)
        
    x_l1, x_l2, x_l3 = Lorenz(T, dt=dt, sigma = sigma, rho = rho, beta = beta, noise_level = noise_level, couplings = couplings)


    y = np.concatenate([x_l1,x_l2],axis = 1)

    N_t = int(T//dt)
    

    col= ["ID","Time"] + [f"Value_{i}" for i in range(1,7)] + [f"Mask_{i}" for i in range(1,7)]   
    
    #df = pd.DataFrame(columns=col)

    num_samples = int(sample_rate * T)

    sample_times = np.random.choice(N_t,num_samples, replace = False)
    samples = y[sample_times,:]

    # Now only select some of the samples.
    mask = np.ones_like(samples)
    
    random_mat = np.random.uniform(size = samples.shape)

    mask[random_mat>multiple_sample_rate] = 0
    samples[random_mat>multiple_sample_rate] = 0

    del random_mat

    samples = samples[mask.sum(axis=1)>0]
    sample_times = sample_times[mask.sum(axis=1)>0]
    mask    = mask[mask.sum(axis=1)>0]
    
    sample_times = sample_times*dt

    num_samples = samples.shape[0]

    if num_series > 1 :
        bins = np.linspace(0, T, num_series+1)
        id_vec = np.expand_dims(np.digitize(sample_times,bins),1)
        sample_times = sample_times - bins[id_vec-1][:,0]        
    else:
        id_vec = np.ones((num_samples,1))


    df = pd.DataFrame(np.concatenate((id_vec,np.expand_dims(sample_times,1),samples,mask),1),columns=col)
     
    df.reset_index(drop=True,inplace=True)
    return(df,y)

def generate_Lorenz_data(seed=0):
    sampling_not_at_random = True

    #Data generation
    
    seed = 421 + seed

    T = 10000
    dt = 0.01
    sigma = 10
    rho = 28
    beta = 8/3
    noise_level = 0.
    couplings = [0,3.5]
    sample_rate = 30
    multiple_sample_rate = 0.3
    num_series = 1000

    df,y = Lorenz_sample(T = T, dt = dt, sigma = sigma, rho = rho, beta = beta, noise_level = noise_level, couplings = couplings, sample_rate = sample_rate, multiple_sample_rate = multiple_sample_rate, num_series = num_series)
    
    df,y = scaling(df,y)

    #Save metadata dictionary
    metadata_dict = {"T":T, "delta_t":dt, "rho": rho,
                    "sigma" : sigma, "beta": beta, "noise_level" : noise_level,
                    "couplings" : couplings, "num_series" : num_series,
                    "sample_rate": sample_rate, "multiple_sample_rate": multiple_sample_rate}

    return df,y, metadata_dict


def Double_pendulum(T, dt, l1, l2, m1, m2, noise_level):
    N_t = int(T//dt) #number of timepoint
    state = np.zeros((N_t,4))
    system = DPendulum(l1=l1, l2=l2, m1=m1, m2=m2, theta1 = -1)
    for i in range(N_t):
        state[i,0] = system.theta1
        state[i,1] = system.theta2
        state[i,2] = system.p1
        state[i,3] = system.p2
        system.leapfrog_step(dt)

    return state

def coupled_chaotic_pendulum(T, dt, l_p1, l_p2, l_p3, A_p1, A_p2, A_p3,  c_12, c_21,c_13, c_23, omega_p1, omega_p2, omega_p3, q_p1, q_p2, q_p3, theta):
    N_t = int(T//dt)
    state_1 = np.zeros((N_t, 2))
    state_2 = np.zeros((N_t, 2))
    state_3 = np.zeros((N_t, 2))
    systemA = DampedDrivenPendulum(l=l_p1, forcing_amplitude = A_p1, forcing_omega = omega_p1, q=q_p1)
    systemA.state[0] = theta + 0.05 * np.random.randn()
    systemB = DampedDrivenPendulum(l=l_p2, forcing_amplitude = A_p2, forcing_omega = omega_p2, q=q_p2)
    systemB.state[0] = theta + 0.05 * np.random.randn()
    systemC = DampedDrivenPendulum(l=l_p3, forcing_amplitude = A_p3, forcing_omega = omega_p3, q=q_p3)
    systemC.state[0] = theta + 0.05 * np.random.randn()

    systemA.couple(systemB,c_21)
    systemB.couple(systemA,c_12)
    systemA.couple(systemC,c_13)
    systemB.couple(systemC,c_23)
    tau = 3e-3

    for i in range(N_t):
        state_1[i,:] = systemA.state
        state_2[i,:] = systemB.state
        state_3[i,:] = systemC.state
        systemA.step(dt)
        systemB.step(dt)
        if ((c_13!=0) or (c_23!=0)):
            systemC.step(dt)
   
    state_1[:,0] = np.sin(state_1[:,0])
    state_2[:,0] = np.sin(state_2[:,0])
    state_3[:,0] = np.sin(state_3[:,0])

    return state_1, state_2, state_3

def coupled_double_pendulum(T, dt, l_p1, l_p2, l_p3, m_p1, m_p2, m_p3, c_12, c_21,c_31, c_32, theta1, theta2, noise_level):
    N_t = int(T//dt)
    state_1 = np.zeros((N_t, 4))
    state_2 = np.zeros((N_t, 4))
    state_3 = np.zeros((N_t,4))
    systemA = DPendulum(l1=l_p1[0], l2=l_p1[1], m1=m_p1[0], m2=m_p1[1], theta1 = theta1, theta2 = theta2)
    systemB = DPendulum(l1=l_p2[0], l2=l_p2[1], m1=m_p2[0], m2=m_p2[1], theta1 = theta1, theta2 = theta2)
    systemC = DPendulum(l1=l_p3[0], l2=l_p3[1], m1=m_p3[0], m2=m_p3[1], theta1 = theta1, theta2 = theta2)
    #systemB.theta1 += 0.5
    systemA.couple(systemB,c_21)
    systemB.couple(systemA,c_12)
    systemA.couple(systemC,c_31)
    systemB.couple(systemC,c_32)
    tau = 3e-3
    for i in range(N_t):
        state_1[i,0] = systemA.theta1
        state_1[i,1] = systemA.theta2
        state_1[i,2] = systemA.p1
        state_1[i,3] = systemA.p2
        state_2[i,0] = systemB.theta1
        state_2[i,1] = systemB.theta2
        state_2[i,2] = systemB.p1
        state_2[i,3] = systemB.p2
        state_3[i,0] = systemC.theta1
        state_3[i,1] = systemC.theta2
        state_3[i,2] = systemC.p1
        state_3[i,3] = systemC.p2
        systemA.leapfrog_step(dt)
        systemB.leapfrog_step(dt)
        if ((c_31!=0) or (c_32!=0)):
            systemC.leapfrog_step(dt)
    
    return state_1, state_2, state_3




def Double_pendulum_sample(T, dt, l1,l2,m1,m2, noise_level, sample_rate, multiple_sample_rate, num_series=1, seed = 432):
    '''
    Sample from the double pendulum (theta1, theta2, p1, p2)
    The sample rate should be expressed in samples per unit of time. (on average there will be sample_rate*T sample per series) - exactly :D
    The multiple_sample_rate gives the proportion of samples wich are jointly sampled (for all dimensions)
    '''
    np.random.seed(seed)

    state = Double_pendulum(T = T, dt = dt, l1 = l1, l2 = l2, m1 = m1, m2 = m2, noise_level = noise_level)

    y = state
    y += noise_level * np.random.randn(y.shape)

    N_t = int(T//dt)

    col = ["ID", "Time"] + [f"Value_{i}" for i in range(1,5)] + [f"Mask_{i}" for i in range(1,5)]

    num_samples = int(sample_rate * T)
    sample_times = np.random.choice(N_t, num_samples, replace = False)
    samples = y[sample_times,:]

    #select observations
    mask = np.ones_like(samples)
    random_mat = np.random.uniform(size = samples.shape)
    mask[random_mat>multiple_sample_rate] = 0
    samples[random_mat>multiple_sample_rate] = 0
    del random_mat

    samples = samples[mask.sum(axis=1)>0]
    sample_times = sample_times[mask.sum(axis=1)>0]
    mask    = mask[mask.sum(axis=1)>0]
    
    sample_times = sample_times*dt

    num_samples = samples.shape[0]

    if num_series > 1 :
        bins = np.linspace(0, T, num_series+1)
        id_vec = np.expand_dims(np.digitize(sample_times,bins),1)
        sample_times = sample_times - bins[id_vec-1][:,0]        
    else:
        id_vec = np.ones((num_samples,1))


    df = pd.DataFrame(np.concatenate((id_vec,np.expand_dims(sample_times,1),samples,mask),1),columns=col)
     
    df.reset_index(drop=True,inplace=True)
    return(df,y)

def Coupled_Chaotic_pendulum_sample(T, dt, l_p1, l_p2, l_p3, A_p1, A_p2, A_p3, omega_p1, omega_p2, omega_p3, q_p1, q_p2, q_p3, c_12,c_21,c_13,c_23, noise_level, sample_rate, multiple_sample_rate, theta = 0.0, num_series=1, seed = 432, sampling_not_at_random=False):
    '''
    Sample from the chaotic pendulum (theta, p1)
    The sample rate should be expressed in samples per unit of time. (on average there will be sample_rate*T sample per series) - exactly :D
    The multiple_sample_rate gives the proportion of samples wich are jointly sampled (for all dimensions)
    '''
    np.random.seed(seed)

    state_1, state_2, state_3 =  coupled_chaotic_pendulum(T = T, dt = dt, l_p1 = l_p1, l_p2 = l_p2, l_p3 = l_p3, A_p1 = A_p1, A_p2 = A_p2, A_p3 = A_p3, omega_p1 = omega_p1, omega_p2 = omega_p2, omega_p3 = omega_p3, q_p1 = q_p1, q_p2 = q_p2, q_p3 = q_p3, theta = theta, c_12 = c_12, c_21 = c_21, c_13 = c_13, c_23 = c_23)

    y1 = state_1
    y2 = state_2
    y3 = state_3
    y1 += noise_level * np.random.randn(*y1.shape)
    y2 += noise_level * np.random.randn(*y2.shape)
    y3 += noise_level * np.random.randn(*y3.shape)

    N_t = int(T//dt)

    ndim = y1.shape[1]
    col = ["ID", "Time"] + [f"Value_{i}" for i in range(1,ndim+1)] + [f"Mask_{i}" for i in range(1,ndim+1)]

    ys = [y1, y2, y3]
    dfs = []
    for i in range(len(ys)):
        
        num_samples = int(sample_rate * T)
        p = np.ones(N_t)/N_t
        if sampling_not_at_random:
            p[np.where(np.abs(np.sin(ys[i][:,0]))<(np.sin(np.pi/4)))[0]] = 1.5/N_t
            p = p/p.sum()

        sample_times = np.random.choice(N_t, num_samples, replace = False, p = p)
        samples = ys[i][sample_times,:]
        

        #select observations
        mask = np.ones_like(samples)
        random_mat = np.random.uniform(size = samples.shape)
        mask[random_mat>multiple_sample_rate] = 0
        samples[random_mat>multiple_sample_rate] = 0
        del random_mat

        samples = samples[mask.sum(axis=1)>0]
        sample_times = sample_times[mask.sum(axis=1)>0]
        mask    = mask[mask.sum(axis=1)>0]
        
        sample_times = sample_times*dt

        num_samples = samples.shape[0]

        if num_series > 1 :
            bins = np.linspace(0, T, num_series+1)
            id_vec = np.expand_dims(np.digitize(sample_times,bins),1)
            sample_times = sample_times - bins[id_vec-1][:,0]        
        else:
            id_vec = np.ones((num_samples,1))


        df = pd.DataFrame(np.concatenate((id_vec,np.expand_dims(sample_times,1),samples,mask),1),columns=col)
         
        df.reset_index(drop=True,inplace=True)

        dfs += [df]
    
    return(dfs,ys)

def Coupled_Double_pendulum_sample(T, dt, l_p1,l_p2,l_p3,m_p1,m_p2,m_p3,c_12,c_21,c_31,c_32, noise_level, sample_rate, multiple_sample_rate, theta1 = -1, theta2 = 0.5, num_series=1, seed = 432, sampling_not_at_random = False):
    '''
    Sample from the double pendulum (theta1, theta2, p1, p2)
    The sample rate should be expressed in samples per unit of time. (on average there will be sample_rate*T sample per series) - exactly :D
    The multiple_sample_rate gives the proportion of samples wich are jointly sampled (for all dimensions)
    '''
    np.random.seed(seed)

    state_1, state_2, state_3 = coupled_double_pendulum(T = T, dt = dt, l_p1 = l_p1, l_p2 = l_p2, l_p3 = l_p3, m_p1 = m_p1, m_p2 = m_p2, m_p3 = m_p3, c_12 = c_12, c_21 = c_21, c_31 = c_31, c_32 = c_32, theta1 = theta1, theta2 =theta2, noise_level = noise_level)

    y1 = state_1
    y2 = state_2
    y3 = state_3
    y1 += noise_level * np.random.randn(*y1.shape)
    y2 += noise_level * np.random.randn(*y2.shape)
    y3 += noise_level * np.random.randn(*y3.shape)

    N_t = int(T//dt)

    col = ["ID", "Time"] + [f"Value_{i}" for i in range(1,5)] + [f"Mask_{i}" for i in range(1,5)]

    ys = [y1, y2, y3]
    dfs = []
    for i in range(len(ys)):
        
        num_samples = int(sample_rate * T)

        p = np.ones(N_t)/N_t
        if sampling_not_at_random:
            p[np.where(np.abs(np.sin(ys[i][:,0]))>(np.sin(np.pi/4)))[0]] = 2/N_t
            p = p/p.sum()
        
        sample_times = np.random.choice(N_t, num_samples, replace = False,p=p)
        samples = ys[i][sample_times,:]
        

        #select observations
        mask = np.ones_like(samples)
        random_mat = np.random.uniform(size = samples.shape)
        mask[random_mat>multiple_sample_rate] = 0
        samples[random_mat>multiple_sample_rate] = 0
        del random_mat

        samples = samples[mask.sum(axis=1)>0]
        sample_times = sample_times[mask.sum(axis=1)>0]
        mask    = mask[mask.sum(axis=1)>0]
        
        sample_times = sample_times*dt

        num_samples = samples.shape[0]

        if num_series > 1 :
            bins = np.linspace(0, T, num_series+1)
            id_vec = np.expand_dims(np.digitize(sample_times,bins),1)
            sample_times = sample_times - bins[id_vec-1][:,0]        
        else:
            id_vec = np.ones((num_samples,1))


        df = pd.DataFrame(np.concatenate((id_vec,np.expand_dims(sample_times,1),samples,mask),1),columns=col)
         
        df.reset_index(drop=True,inplace=True)

        dfs += [df]
    
    return(dfs,ys)

def scaling(df,y):
    val_cols = [c for c in df.columns if "Value" in c] 
    mask_cols = [c for c in df.columns if "Mask" in c]

    for i in range(len(val_cols)):
        m = df.loc[df[mask_cols[i]]==1,val_cols[i]].mean()
        s = df.loc[df[mask_cols[i]]==1,val_cols[i]].std()
        df.loc[df[mask_cols[i]]==1,val_cols[i]] -= m
        df.loc[df[mask_cols[i]]==1,val_cols[i]] /= s
        y[:,i] = (y[:,i]-m)/s
    
    return(df,y)



if __name__=="__main__":

    T = 10000
    dt = 0.003
    l1 = 1.0
    l2 = 1.0
    m1 = 2.0
    m2 = 1.0
    noise_level = 0.
#    couplings = [0,0]
    sample_rate = 50 #10
    multiple_sample_rate = 0.6
    num_series = 1000

    df,y = Double_pendulum_sample(T = T, dt = dt, l1 = l1, l2 = l2, m1 = m1, m2 = m2, noise_level = noise_level, sample_rate = sample_rate, multiple_sample_rate = multiple_sample_rate, num_series = num_series)


    df1,y_s = scaling(df,y)
    #df2 = scaling(dfs[1])


    #Save metadata dictionary
    metadata_dict = {"T":T, "delta_t":dt, "l1": l1, "l2": l2,
                    "m1" : m1, "m2": m2, "noise_level" : noise_level,
                    "num_series" : num_series,
                    "sample_rate": sample_rate, "multiple_sample_rate": multiple_sample_rate}
    np.save(f"DPendulum_metadata.npy",metadata_dict)

    df.to_csv("DPendulum_data.csv", index = False)
    #Plot some examples and store them.
    import os
    N_examples = 10
    examples_dir = f"Dpendulum_paths_examples/"
    if not os.path.exists(examples_dir):
        os.makedirs(examples_dir)
    for ex in range(N_examples):
        idx = np.random.randint(low=0,high=df["ID"].nunique())
        plt.figure()
        print(idx)
        for dim in [0,1,2,3]:
            random_sample = df.loc[df["ID"]==idx].sort_values(by="Time").values
            obs_mask = random_sample[:,2+4+dim]==1
            plt.scatter(random_sample[obs_mask,1],random_sample[obs_mask,2+dim])
            plt.title("Example of generated trajectory")
            plt.xlabel("Time")
        #plt.savefig(f"{examples_dir}{args.prefix}_{ex}.pdf")
        plt.savefig(f"{examples_dir}full_example_{ex}.pdf")
        plt.close()

def compress_df(df_in,fact,time_bin = 10):
    df = df_in.copy()
    df["IDbis"] = (df.ID-1) % fact
    df.Time = df.Time + time_bin * df.IDbis
    df.ID = df.ID - df.IDbis
    df.ID = df.ID.map(dict(zip(df.ID.unique(),np.arange(df.ID.nunique()))))
    df.drop("IDbis", axis = 1,inplace = True)
    return df

def compress_and_shuffle_df(df_in, fact, time_bin = 10):
    df_list = []
    df = df_in.copy()
    id_max = 0
    for i in range(fact):
        df_ = df.loc[(df.ID>i)&(df.ID<(df.ID.max()-i))].copy()
        df_["ID"] = df_.ID-i
        df_c = compress_df(df_,fact,time_bin)
        df_c["ID"] = df_c.ID+id_max
        df_list.append(df_c)
        id_max = df_c.ID.max()+1
    return pd.concat(df_list)

