import numpy as np
from numba import njit, float64

@njit(float64[:](float64[:],float64), fastmath=True)
def ricampionamento(array_oss, bin):
    sample=[]
    for _ in range(int(len(array_oss)/bin)+1):
        ii=np.random.randint(0, len(array_oss)+1)
        sample.extend(array_oss[ii:min(ii+bin, len(array_oss))])
    return np.array(sample)

def bootstrap_binning(array_osservabile, bin):
    obs_list=[]
    ritorno=[]
    array_osservabile=(array_osservabile.astype(np.float64))
    for _ in range(200):
        sampler=ricampionamento(array_osservabile, bin)
        sampler=np.array(sampler)**2
        obs=np.mean(sampler)
        obs_list.append(obs)
    obl_list2 = np.array(obs_list)**2
    # ritorno.append(np.std(obs_list))
    # ritorno = np.array(ritorno)
    ritorno.append(np.sqrt(np.mean(obl_list2)-np.mean(obs_list)**2))
    return ritorno

