import numpy as np
from numba import njit

@njit(fastmath=True)
def ricampionamento(array_osservabile, bin):
    sample=[]
    for _ in range(int(len(array_osservabile)/bin)):
        ii=np.random.randint(0, len(array_osservabile)+1)
        sample.extend(array_osservabile[ii:min(ii+bin, len(array_osservabile))]) 
    return np.array(sample)

def bootstrap_binning(array_osservabile, bin):
    obs_list=[]
    for _ in range(100):
        sample=ricampionamento(array_osservabile, bin)
        obs=np.var(sample)
        obs_list.append(obs)
    ritorno=np.std(obs_list)
    return ritorno

def sigma(q):
    q=np.array(q)
    q4=np.mean(q**4)
    q2=np.mean(q**2)
    sigma=(q4-q2**2)**0.5
    return(sigma)