import numpy as np
from numba import njit, float64

for _ in range(200,550,50):
    a = np.loadtxt(f"Results/Local_MonteCarlo/beta2_Nt_500_delta=0.5/Q_Nt={_}_beta=2_eta={(2./_):.5f}_delta=0.5.txt")
    Q2 = np.mean(a**2)
    Q4 = np.mean(a**4)
    var = Q4-Q2**2
    print(var)

@njit(float64[:](float64[:],float64), fastmath=True)
def ricampionamento(array_oss, bin):
    sample=[]
    for _ in range(int(len(array_oss)/bin)+1):
        ii=np.random.randint(0, len(array_oss)+1)
        sample.extend(array_oss[ii:min(ii+bin, len(array_oss))])
    return np.array(sample)

def bootstrap_binning(array_osservabile, bin):
    mediaQ2=[]
    mediaQ4=[]
    ritorno=[]
    for _ in range(10):
        sampler=ricampionamento(array_osservabile, bin)
        mediaQ2.append(np.mean(sampler**2))
        mediaQ4.append(np.mean(sampler**4))
    mediaQ2 = np.array(mediaQ2)
    mediaQ4 = np.array(mediaQ4)
    ritorno.append((mediaQ4-(mediaQ2)**2))
    ritorno = np.array(ritorno)
    ritorno= np.std(ritorno)
    return (ritorno)

def bootstrap(array_osservabile, bin):
    sigma1 = []
    osservabile=[]
    for _ in range(10):
        sample=ricampionamento(array_osservabile, bin)
        osservabile2 = sample**2 
        sigma1.append(np.std(osservabile2))
    sigma1 = np.array(sigma1)
    return max(sigma1)

# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(np.arange(200,550,50), tau)
# plt.show()