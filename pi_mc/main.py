import argparse
from functools import partial
import logging
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import Module.bootstrap as boot
import Model.constants as const
import Module.func_cerchio as fnc
import Module.correlation as corr
from Module.makedir import smart_makedir as mk

Beta = const.BETA
Nt = const.NT
a = const.ETA
Nt_arr = const.NT_ARRAY
a_arr = const.ETA_ARRAY
delta = const.DELTA_METRO
cammini = const.PATHS
term = const.TERM
bin_arr = const.BIN_ARRAY

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PIMC')
    parser.add_argument("-l", "--log", default="info", help=("Provide logging level. Example --log debug', default='info"))
    parser.add_argument('-s','--save', action='store_true', help='Replace everything in the corrisponding folder')
    args = parser.parse_args()

    levels = {'critical': logging.CRITICAL,
              'error': logging.ERROR,
              'warning': logging.WARNING,
              'info': logging.INFO,
              'debug': logging.DEBUG}

    logging.basicConfig(level=levels[args.log])

    new_measure = args.save
    if new_measure:
        mk("Results")
        mk("MonteCarlo_History")
        mk("Distribution")
        # mk("Graphs")

    local_montecarlo=True
    global_montecarlo = True
    bootstrap_exe = False

    if local_montecarlo:
        processi = len(Nt_arr)
        mk(f"Results/Local_MonteCarlo/beta{Beta}_delta{delta}_{Nt_arr[-1]}")
        mk(f"MonteCarlo_History/Local_MonteCarlo/beta{Beta}_delta{delta}_{Nt_arr[-1]}")
        mk(f"Distribution/Local_MonteCarlo/beta{Beta}_delta{delta}_{Nt_arr[-1]}")
        path = f"MonteCarlo_History/Local_MonteCarlo/beta{Beta}_delta{delta}_{Nt_arr[-1]}"
        path2 = f"Distribution/Local_MonteCarlo/beta{Beta}_delta{delta}_{Nt_arr[-1]}"
        with multiprocessing.Pool(processes=processi) as pool:
            altri = partial(fnc.cammino_piano, Beta, cammini, term, delta)
            q_arr = np.array(pool.map(altri, Nt_arr), dtype="object")
            pool.close()
            pool.join()
        for Nt in range(len(Nt_arr)):
            a=Beta/Nt_arr[Nt]
            q = q_arr[Nt, :]
            np.savetxt(f"Results/Local_MonteCarlo/beta{Beta}_delta{delta}_{Nt_arr[-1]}/Q_Nt={Nt_arr[Nt]}_beta={Beta}_eta={a:.5f}_delta={delta}.txt", q, fmt='%.1f')
            fnc.graph_montecarlo(Nt_arr, Nt, Beta, q, path, delta)
            fnc.distr_q(Nt_arr, Nt, Beta, q, path2, delta)

    if global_montecarlo:
        processi = len(Nt_arr)
        mk(f"Results/Global_MonteCarlo/beta{Beta}_delta{delta}_{Nt_arr[-1]}")
        mk(f"MonteCarlo_History/Global_MonteCarlo/beta{Beta}_delta{delta}_{Nt_arr[-1]}")
        mk(f"Distribution/Global_MonteCarlo/beta{Beta}_delta{delta}_{Nt_arr[-1]}")
        path = f"MonteCarlo_History/Global_MonteCarlo/beta{Beta}_delta{delta}_{Nt_arr[-1]}"
        path2 = f"Distribution/Global_MonteCarlo/beta{Beta}_delta{delta}_{Nt_arr[-1]}"
        with multiprocessing.Pool(processes=processi) as pool:
            altri = partial(fnc.Tailor, Beta, cammini, term, delta)
            q_tailor=np.array(pool.map(altri, Nt_arr), dtype="object")
            pool.close()
            pool.join()
        for Nt in range(len(Nt_arr)):
            a=Beta/Nt_arr[Nt]
            q = q_tailor[Nt, :]
            np.savetxt(f"Results/Global_MonteCarlo/beta{Beta}_delta{delta}_{Nt_arr[-1]}/q_Nt_{Nt_arr[Nt]}_beta={Beta}_eta={a:.5f}_delta={delta}_tailor.txt", q, fmt='%.1f')
            fnc.graph_montecarlo(Nt_arr, Nt, Beta, q, path, delta)
            fnc.distr_q(Nt_arr, Nt, Beta, q, path2, delta)

#---

    # tau_list = []
    # if bootstrap_exe == True:
    #     sigma=[]
    #     if global_montecarlo == True : q_arr=q_tailor  
    #     proc=len(bin_arr)
    #     num_Nt=len(Nt_arr)
    #     for qq in range(num_Nt):
    #         logging.info(f'siamo al {Nt_arr[qq]} Nt')
    #         Q=q_arr[qq,:]
    #         with multiprocessing.Pool(processes=proc) as pool:
    #             parziale=partial(boot.bootstrap_binning, Q)
    #             results=np.array(pool.map(parziale, bin_arr), dtype='object')
    #             pool.close()
    #             pool.join()
    #             sigma.append(max(results))

    #     sigma2_naive=[]
    #     N=len(q_arr[0,:])
    #     for i in range(processi):
    #         q2=np.array(q_arr[i, :]**2)
    #         q4=np.array(q2)**2
    #         sigma2_naive.append((np.mean(q4)-np.mean(q2)**2))

    #     sigma2_naive=np.array(sigma2_naive)
        
    #     print('lungo', N)
    #     tau=np.array([(0.5*N*(sigma[t]**2))/(sigma2_naive[t]) for t in range(len(sigma))])
    #     plt.scatter(Nt_arr, tau, s=4, c='red')
    #     plt.show()

    # q2=np.array(q_arr[0,:])**2
    # C=corr.connected_correlation_time(q2)
    # tau=corr.correlation_time(q2)
    # tau_list=[]
    # for i in range(processi):
    #     q2=np.array((q_arr[i, :]))**2
    #     tau_list.append(corr.correlation_time(q2))
    # print(tau_list)
    # plt.scatter(a_arr, tau_list)
    # plt.show()
