import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import module.func_cerchio as fnc
import multiprocessing
from functools import partial
import module.constants as const
import module.bootstrap as boot

cammini = const.CAMMINI
Tailor = const.TAILOR  # bool
bootstrap_exe = const.BOOTSTRAP  # bool
bin_arr = const.BIN_ARRAY
            
if __name__ == "__main__":
    cammini = 100000
    term = 1000
    Nt_arr = np.arange(500, 6000, 500)
    tau_list = []
    processi = len(Nt_arr)

    if Tailor == False:
        with multiprocessing.Pool(processes=processi) as pool:
            q_arr = np.array(
                pool.map(fnc.cammino_piano, Nt_arr), dtype="object")
            pool.close()
            pool.join()
        for Nt in range(len(Nt_arr)):
            q = q_arr[Nt, 0]
            plt.plot(range(len(q)), q)
            plt.show()

    if Tailor == True:
        q=0
        with multiprocessing.Pool(processes=processi) as pool:
            q_tailor=np.array(pool.map(fnc.Tailor_exe, Nt_arr), dtype='object')
            pool.close()
            pool.join()
        for Nt in range(len(Nt_arr)):
            q = q_tailor[Nt, 0]
            plt.plot(range(len(q)), q)
            plt.show()

    # if bootstrap_exe == True:
        


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


