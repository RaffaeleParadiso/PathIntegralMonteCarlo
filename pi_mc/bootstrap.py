import numpy as np
import module.bootstrap as bt

obs_var = []
obs_naive = []
for i in range(0, 8):
    load = np.loadtxt(f"results/q_Nt_{i}")
    obs_naive.append(np.var(load))
    bin = [10,20,40,80,160,320]
    s = []
    for i in range(6):
        omax = bt.bootstrap_binning(load, bin[i])
        s.append(omax)
    obs_var.append(max(s))
obs_var = np.array(obs_var)
obs_naive = np.array(obs_naive)
tau=obs_var/(2*obs_naive)
import matplotlib.pyplot as plt
plt.figure()
plt.scatter(np.arange(len(tau)), tau)
plt.show()


