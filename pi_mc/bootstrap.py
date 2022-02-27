from functools import partial
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
from numba import njit, float64
import Model.constants as const
from Module.makedir import smart_makedir as mk
from scipy.optimize import curve_fit

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
    for _ in range(50):
        sampler=ricampionamento(array_osservabile, bin)
        obs_list.append(np.mean(sampler**2))
    obs_list = np.array(obs_list)
    ritorno.append(np.var(obs_list))
    # ritorno.append(np.std(obs_list))
    ritorno = np.array(ritorno)
    # ritorno.append((np.mean(obs_list**2)-np.mean(obs_list)**2))
    return ritorno

def expo(x, a, b):
    return a*np.exp(-x/b)

Beta = const.BETA
Nt_arr = const.NT_ARRAY
a_arr = const.ETA_ARRAY
delta = const.DELTA_METRO
cammini = const.PATHS
term = const.TERM
bin_arr = const.BIN_ARRAY
tau_list = []
sigma=[]
sigma2_naive=[]
# mk("Graphs_tau")
x = -1 # freeze
for Nt in range(200,550,50):
    q_arr = (np.loadtxt(f"Results/Local_MonteCarlo/beta2_Nt_550_delta=0.5/" \
        f"Q_Nt={Nt}_beta=2_eta={(Beta/Nt):.5f}_delta=0.5.txt"))
        # _delta={np.sqrt(Beta/Nt_arr[Nt]):.5f}.txt"))
    print(f'siamo al {Nt} Nt')
    Q=q_arr
    with multiprocessing.Pool(processes=len(bin_arr)) as pool:
        parziale=partial(bootstrap_binning, Q)
        results=np.array(pool.map(parziale, bin_arr), dtype='object')
        pool.close()
        pool.join()
        sigma.append(max(results))

    N=len(q_arr)
    for i in range(len(Nt_arr)):
        q2=np.array(q_arr**2)
        sigma2_naive.append((np.mean(q2**2)-np.mean(q2)**2))
s = len(sigma)
tau=np.array([(0.5*N*(sigma[t]))/(sigma2_naive[t]) for t in range(s)])
np.savetxt("tau.txt", tau)
tau2 = np.ndarray.flatten(tau)
print(tau)

xa = np.arange(200,550,50)
popt, pcov = curve_fit(expo, xa, tau2, p0=(0.07,0.029))
a, ta = popt 
print(a)
print(ta)
print('lungo', N)
plt.figure()
plt.plot(xa, expo(xa, *popt), 'r-', label="Fitted Curve")
plt.legend()
plt.scatter(Nt_arr[:x], tau, c='red')
plt.xticks(Nt_arr[:x])
plt.xlabel("Nt")
plt.ylabel(rf"$\tau$")
plt.yscale("log")
plt.savefig("Graphs_tau/tau.png")
plt.show()
