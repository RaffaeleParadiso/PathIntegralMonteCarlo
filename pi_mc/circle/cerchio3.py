from unittest import result
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import module.func_cerchio as fnc
import multiprocessing
from functools import partial
import module.constants as const
import module.bootstrap as boot
from scipy.stats import norm
import logging

cammini = const.CAMMINI
Tailor = const.TAILOR  # bool
bootstrap_exe = const.BOOTSTRAP  # bool
bin_arr = const.BIN_ARRAY
logging.basicConfig(level=logging.INFO)
            
if __name__ == "__main__":
    cammini = 100000
    term = 100
    Nt_arr = np.arange(100, 600, 50)
    tau_list = []
    processi = len(Nt_arr)
    Tailor=False

    if Tailor == False:
        with multiprocessing.Pool(processes=processi) as pool:
            q_arr = np.array(
                pool.map(fnc.cammino_piano, Nt_arr), dtype="object")
            pool.close()
            pool.join()
        for Nt in range(len(Nt_arr)):
            a=2./Nt_arr[Nt]
            q = q_arr[Nt, :]            
            plt.plot(range(len(q)), q)
            plt.show()
            bins=np.arange(q.min(), q.max()+2)
            bins=bins-0.5
            xlims=[-15,15]
            x=np.linspace(*xlims, 1000)
            plt.figure()
            plt.hist(q, bins, fill=False)
            plt.xlim(xlims)
            plt.plot(x, norm.pdf(x, 0, np.sqrt(a)))
            plt.show()
        print(q_arr[0,:])
        print(len(q_arr[0,:]), len(q_arr[:,0]))
        q2=np.array([np.mean(np.array(q_arr[k,:])**2) for k in range(len(Nt_arr))])
        print('chest è',q2)

    if Tailor == True:
        with multiprocessing.Pool(processes=processi) as pool:
            q_tailor=np.array(pool.map(fnc.Tailor_exe, Nt_arr), dtype='object')
            pool.close()
            pool.join()
            print('q tailor', len(q_tailor[0, :]))
        for Nt in range(len(Nt_arr)):
            q = q_tailor[Nt, :]
            plt.plot(range(len(q)), q)
            plt.show()
    if bootstrap_exe == True:
        sigma=[]
        if Tailor == False:
            proc=len(bin_arr)
            num_Nt=len(Nt_arr)
            for qq in range(num_Nt):
                logging.info(f'siamo al {Nt_arr[qq]} Nt')
                Q=q_arr[qq,:]
                with multiprocessing.Pool(processes=proc) as pool:
                    parziale=partial(boot.bootstrap_binning, Q)
                    results=np.array(pool.map(parziale, bin_arr), dtype='object')
                    pool.close()
                    pool.join()
                    sigma.append(max(results))


        # sigma_naive=np.array([np.std(np.array(q_arr[p,:])) for p in range(processi)])
        # print('sigma boot',sigma)
        # print('sigma naive', sigma_naive)
        # N=len(q_arr[0,:])
        # tau=np.array([(0.5*N*sigma[t]**2)/(sigma_naive[t]**2) for t in range(len(sigma))])
        # plt.scatter(range(len(tau)), tau, s=4, c='red')
        # plt.show()
        tau=0
       






        


        # if Tailor == True:
        #     q2_bootstrap = np.zeros((len(q_tailor[0, 0]), len(Nt_arr)))
        #     for i in range(len(Nt_arr)):
        #         q2_bootstrap[::, i] = np.array(q_tailor[i, 0])**2

        # if Tailor == False:
        #     q2_bootstrap = np.zeros((len(q_arr[0, 0]), len(Nt_arr)))
        #     for i in range(len(Nt_arr)):
        #         q2_bootstrap[::, i] = np.array(q_arr[i, 0])**2
        #         q2_mean.append(np.mean(q2_bootstrap[::,i]))
        #         vari.append(np.var(q2_bootstrap[::,i]))
        #     print(q2_mean)
        #     print('chest so i varianz', vari)
        # long = len(q2_bootstrap[0, ::]) # long= numero di array q2 = al numero di Nt considerati
        # number_bins = len(bin_arr)
        # for qq in range(long):
        #     with multiprocessing.Pool(processes=number_bins) as pool:
        #         partial_boot = partial(
        #             boot.bootstrap_binning, q2_bootstrap[::, qq])
        #         results_boot = np.array(
        #             pool.map(partial_boot, bin_arr), dtype="object")
        #         sigma_max = max(results_boot)
        #         pool.close()
        #         pool.join()
        #         sigma_q2_max.append(sigma_max)

