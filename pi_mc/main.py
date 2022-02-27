import argparse
from functools import partial
import multiprocessing
import numpy as np
import Model.constants as const
import Module.func_cerchio as fnc
from Module.makedir import smart_makedir as mk

Beta = const.BETA
Nt_arr = const.NT_ARRAY
a_arr = const.ETA_ARRAY
delta_l = const.DELTA_L
delta = const.DELTA_METRO
cammini = const.PATHS
term = const.TERM
bin_arr = const.BIN_ARRAY

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PIMC')
    parser.add_argument('-s','--save', action='store_true', help='Replace everything in the corrisponding folder')
    parser.add_argument('-m','--monte', action='store_true', help='Start a new MonteCarlo simulation')
    parser.add_argument('-d','--deltaa', action='store_true', help='Start MonteCarlo simulation with delta 0.5 Default and delta sqrt(a) with flag')

    args = parser.parse_args()

    new_measure = args.save
    dteta = args.deltaa
    if new_measure:
        mk("Results")
        mk("MonteCarlo_History")
        mk("Distribution")
    local_montecarlo=args.monte
    global_montecarlo = args.monte

    processi = len(Nt_arr)
    if local_montecarlo:
        if dteta == True:
            mk(f"Results/Local_MonteCarlo/beta{Beta}_Nt_{Nt_arr[-1]}_delta=sqrt(a)")
            mk(f"MonteCarlo_History/Local_MonteCarlo/beta{Beta}_Nt_{Nt_arr[-1]}_delta=sqrt(a)")
            mk(f"Distribution/Local_MonteCarlo/beta{Beta}_Nt_{Nt_arr[-1]}_delta=sqrt(a)")
            path = f"MonteCarlo_History/Local_MonteCarlo/beta{Beta}_Nt_{Nt_arr[-1]}_delta=sqrt(a)"
            path2 = f"Distribution/Local_MonteCarlo/beta{Beta}_Nt_{Nt_arr[-1]}_delta=sqrt(a)"
        else:
            mk(f"Results/Local_MonteCarlo/beta{Beta}_Nt_{Nt_arr[-1]}_delta=0.5")
            mk(f"MonteCarlo_History/Local_MonteCarlo/beta{Beta}_Nt_{Nt_arr[-1]}_delta=0.5")
            mk(f"Distribution/Local_MonteCarlo/beta{Beta}_Nt_{Nt_arr[-1]}_delta=0.5")
            path = f"MonteCarlo_History/Local_MonteCarlo/beta{Beta}_Nt_{Nt_arr[-1]}_delta=0.5"
            path2 = f"Distribution/Local_MonteCarlo/beta{Beta}_Nt_{Nt_arr[-1]}_delta=0.5"
        with multiprocessing.Pool(processes=processi) as pool:
            altri = partial(fnc.cammino_piano, Beta, cammini, term, delta, dt_eta=dteta)
            q_arr = np.array(pool.map(altri, Nt_arr), dtype="object")
            pool.close()
            pool.join()
        for Nt in range(len(Nt_arr)):
            a=Beta/Nt_arr[Nt]
            q = q_arr[Nt, :]
            if dteta == True:
                np.savetxt(f"Results/Local_MonteCarlo/beta{Beta}_Nt_{Nt_arr[-1]}_delta=sqrt(a)/Q_Nt={Nt_arr[Nt]}_beta={Beta}_eta={a:.5f}_delta={delta_l[Nt]:.5f}.txt", q, fmt='%.1f')
                fnc.graph_montecarlo(Nt_arr, Nt, Beta, q, path, delta_l[Nt])
                fnc.distr_q(Nt_arr, Nt, Beta, q, path2, delta_l[Nt])
            else:
                np.savetxt(f"Results/Local_MonteCarlo/beta{Beta}_Nt_{Nt_arr[-1]}_delta=0.5/Q_Nt={Nt_arr[Nt]}_beta={Beta}_eta={a:.5f}_delta={delta}.txt", q, fmt='%.1f')
                fnc.graph_montecarlo(Nt_arr, Nt, Beta, q, path, delta)
                fnc.distr_q(Nt_arr, Nt, Beta, q, path2, delta)

    if global_montecarlo:
        if dteta == True:
            mk(f"Results/Global_MonteCarlo/beta{Beta}_Nt_{Nt_arr[-1]}_delta=sqrt(a)")
            mk(f"MonteCarlo_History/Global_MonteCarlo/beta{Beta}_Nt_{Nt_arr[-1]}_delta=sqrt(a)")
            mk(f"Distribution/Global_MonteCarlo/beta{Beta}_Nt_{Nt_arr[-1]}_delta=sqrt(a)")
            path = f"MonteCarlo_History/Global_MonteCarlo/beta{Beta}_Nt_{Nt_arr[-1]}_delta=sqrt(a)"
            path2 = f"Distribution/Global_MonteCarlo/beta{Beta}_Nt_{Nt_arr[-1]}_delta=sqrt(a)"
        else:
            mk(f"Results/Global_MonteCarlo/beta{Beta}_Nt_{Nt_arr[-1]}_delta=0.5")
            mk(f"MonteCarlo_History/Global_MonteCarlo/beta{Beta}_Nt_{Nt_arr[-1]}_delta=0.5")
            mk(f"Distribution/Global_MonteCarlo/beta{Beta}_Nt_{Nt_arr[-1]}_delta=0.5")
            path = f"MonteCarlo_History/Global_MonteCarlo/beta{Beta}_Nt_{Nt_arr[-1]}_delta=0.5"
            path2 = f"Distribution/Global_MonteCarlo/beta{Beta}_Nt_{Nt_arr[-1]}_delta=0.5"
        with multiprocessing.Pool(processes=processi) as pool:
            altri = partial(fnc.Tailor, Beta, cammini, term, delta, dt_eta=dteta)
            q_tailor = np.array(pool.map(altri, Nt_arr), dtype="object")
            pool.close()
            pool.join()
        for Nt in range(len(Nt_arr)):
            a=Beta/Nt_arr[Nt]
            q = q_tailor[Nt, :]
            if dteta == True:
                np.savetxt(f"Results/Global_MonteCarlo/beta{Beta}_Nt_{Nt_arr[-1]}_delta=sqrt(a)/Q_Nt={Nt_arr[Nt]}_beta={Beta}_eta={a:.5f}_delta={delta_l[Nt]:.5f}.txt", q, fmt='%.1f')
                fnc.graph_montecarlo(Nt_arr, Nt, Beta, q, path, delta_l[Nt])
                fnc.distr_q(Nt_arr, Nt, Beta, q, path2, delta_l[Nt])
            else:
                np.savetxt(f"Results/Global_MonteCarlo/beta{Beta}_Nt_{Nt_arr[-1]}_delta=0.5/Q_Nt={Nt_arr[Nt]}_beta={Beta}_eta={a:.5f}_delta={delta}.txt", q, fmt='%.1f')
                fnc.graph_montecarlo(Nt_arr, Nt, Beta, q, path, delta)
                fnc.distr_q(Nt_arr, Nt, Beta, q, path2, delta)