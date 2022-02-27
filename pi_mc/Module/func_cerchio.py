import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from scipy.stats import norm
from scipy.optimize import curve_fit

@njit(fastmath=True, cache=True)
def distanza(x, y):
    if (abs(x-y) <= 0.5): d = x-y
    if ((x-y) < -0.5): d = x-y+1.0
    if ((x-y) > 0.5): d = x-y-1.0
    return d

@njit(fastmath=True, cache=True)
def diff_azione(i, y, g, f, yp1):
    d1 = distanza(g[i], yp1)
    d2 = distanza(g[i], y[i])
    d3 = distanza(yp1, f[i])
    d4 = distanza(y[i], f[i])
    return (d1**2)+(d3**2)-(d2**2)-(d4**2)

@njit(fastmath=True, cache=True)
def avvolgimento(Nt, y, g):
    sum = 0.0
    for i in range(0, Nt):
        sum = sum + distanza(g[i], y[i])
    return round(sum)

@njit(fastmath=True, cache=True)
def geometry(Nt):
    npp = [i+1 for i in range(0, Nt)]
    nmm = [i-1 for i in range(0, Nt)]
    npp[Nt-1] = 0
    nmm[0] = Nt-1
    return (npp, nmm)

# @njit(fastmath=True, cache=True)
# def metropolis(y, Nt, a):
#     y_new = y.copy()
#     delta = 0.5
#     ypp, ymm = np.zeros(Nt), np.zeros(Nt)
#     npp, nmm = geometry(Nt)
#     for i in range(0, Nt):
#         r = np.random.random()
#         yprova = (y_new[i]+(1-2*r)*delta)%1
#         ypp[i] = y_new[npp[i]]
#         ymm[i] = y_new[nmm[i]]
#         s = diff_azione(i, y_new, ypp, ymm, yprova)
#         if s <= 0:
#             y_new[i] = yprova
#             y_new[Nt-1] = y_new[0]
#         x = np.random.random()
#         if x < np.exp(-s/(2.0*a)):
#             y_new[i] = yprova
#             y_new[Nt-1] = y_new[0]
#     for i in range(Nt):
#         ypp[i] = y_new[npp[i]]
#         ymm[i] = y_new[nmm[i]]
#     return y_new, ypp

# @njit(fastmath=True, parallel=True)
# def cammino_piano(Beta, cammini, term, Nt):
#     a = Beta/Nt
#     q = []
#     y = np.zeros(shape=(Nt,))
#     # for i in range(Nt):
#     #     y[i] = np.random.random()
#     # y[Nt-1] = y[0]
#     for cam in range(cammini):
#         if cam % 1000 == 0:
#             print('sono al cammino metro', cam)
#         for _ in range(10):
#             y_new, ypp = metropolis(y, Nt, a)
#             y = y_new
#         if cam > term:
#             Q = avvolgimento(Nt, y, ypp)
#             if cam%10==0:
#                 q.append(Q)
#     return q

@njit(fastmath=True, cache=True)
def metropolis(y, Nt, a, ypp, ymm, delta):
    y_new = y.copy()
    for i in range(Nt):
        r = np.random.uniform(-delta, delta)
        yprova = (y_new[i]+r) % 1
        if i == 0:
            ymm[Nt-1] = y_new[0]
        else:
            ymm[i-1] = y_new[i]
        if i == Nt-1:
            ypp[0] = y_new[Nt-1]
        else:
            ypp[i+1] = y_new[i]
        s = diff_azione(i, y_new, ypp, ymm, yprova)
        x = np.exp(-s/(2*a))
        if ((s <= 0) or (np.random.rand() < x)):
            y_new[i] = yprova
        if i == 0:
            ymm[Nt-1] = y_new[0]
        else:
            ymm[i-1] = y_new[i]
        if i == Nt-1:
            ypp[0] = y_new[Nt-1]
        else:
            ypp[i+1] = y_new[i]
    y_new[Nt-1] = y_new[0]
    return y_new, ypp

@njit(fastmath=True, parallel=True)
def cammino_piano(Beta, cammini, term, delt, Nt, dt_eta):
    ypp = np.zeros(Nt)
    ymm = np.zeros(Nt)
    a = (Beta/Nt)
    if dt_eta == True:
        delta = np.sqrt(a)
    else:
        delta = delt
    print(delta)
    q = []
    y=np.zeros(Nt)
    npp, nmm = geometry(Nt)
    for i in range(Nt):
        ypp[i] = y[npp[i]]
        ymm[i] = y[nmm[i]]
    for cam in range(cammini):
        if cam % 10000 == 0:
            print('Cammino metropolis', cam, 'Nt = ', Nt)
        for _ in range(10):
            y_new, ypp = metropolis(y, Nt, a, ypp, ymm, delta)
            y = y_new
        if ((cam > term) and cam % 10 == 0):
            q.append(avvolgimento(Nt, y_new, ypp))
    return np.array(q)

@njit(fastmath=True, cache=True)
def Tailor(Beta, cammini, term, delt, Nt, dt_eta):
    a = (Beta/Nt)
    if dt_eta == True:
        delta = np.sqrt(a)
    else:
        delta = delt
    print(delta)
    p_cut = 0.08
    epsilon = 0.02*a
    y = np.zeros(Nt)
    npp, nmm = geometry(Nt)
    q_list = []
    for t in range(cammini):
        if t % 1000 == 0:
            print('Cammino metropolis', t, 'Nt = ', Nt)
        ypp, ymm = np.zeros(Nt), np.zeros(Nt)
        if np.random.rand() < p_cut:
            y0 = (y[0]+0.5) % 1
            y_new = y.copy()
            for i in range(Nt):
                if abs(distanza(y[i], y0)) <= epsilon:
                    iend = i
                continue
            yprova = (2*y0-y[iend]) % 1
            dS = distanza(yprova, y[iend-1])**2-distanza(y[iend], y[iend-1])**2
            if dS > 1:
                cambio = True
            else:
                if np.random.rand() < np.exp(-dS/(2*a)):
                    cambio = True
                else:
                    cambio = False
            if cambio == True:
                for m in range(iend, Nt):
                    y_new[m] = (2*y0-y[m]) % 1
                    ypp[m] = y_new[npp[m]]
                    ymm[m] = y_new[nmm[m]]
            y = y_new
        for h in range(Nt):
            rand = np.random.randint(0, Nt)
            r = np.random.uniform(-delta, delta)
            y_old = y[rand]
            y_bef = y[nmm[rand]]
            y_aft = y[npp[rand]]
            y_new1 = (y_old+r) % 1
            dS = distanza(y_aft, y_new1)**2+distanza(y_new1, y_bef)**2 - \
                distanza(y_aft, y_old)**2-distanza(y_old, y_bef)**2
            acceptance = np.exp(-dS/(2*a))
            if ((acceptance > 1) or (np.random.rand() < acceptance)):
                y[rand] = y_new1
            ypp[h] = y[npp[h]]
            ymm[h] = y[nmm[h]]
        y[Nt-1] = y[0]
        if ((t > term) and (t % 10 == 0)):
            q_list.append(avvolgimento(Nt, y, ymm))
    return np.array(q_list)

def graph_montecarlo(Nt_arr, Nt, Beta, q, path, delta):
    plt.figure(figsize=(15, 8))
    plt.title(rf"MonteCarlo History for $Nt = {Nt_arr[Nt]}$ and  $\beta = {Beta}$, $\eta = {(Beta/Nt_arr[Nt]):.5f}$, $\delta = {delta:.5f}$")
    plt.ylabel("Q")
    plt.xlabel("MonteCarlo steps")
    plt.plot(range(len(q)), q, label=rf"$Nt = {Nt_arr[Nt]}$")
    # plt.legend(loc=3)
    plt.savefig(rf"{path}/MC_Nt={Nt_arr[Nt]}_beta={Beta}_eta={(Beta/Nt_arr[Nt]):.5f}_delta={delta:.5f}.png", bbox_inches='tight')
    plt.close()

def distr_q(Nt_arr, Nt, Beta, q, path, delta):
    bins=np.arange(q.min(), q.max()+2)
    bins=bins-0.5
    xlims=[-15,15]
    plt.figure(figsize=(6, 6))
    plt.title(rf"Distribution for $Q$, $Nt = {Nt_arr[Nt]}$, $\beta = {Beta}$, $\eta = {(Beta/Nt_arr[Nt]):.5f}$, $\delta = {delta:.5f}$")
    plt.hist(q, bins, density=True,
            histtype='bar', fill=False,
            color = "r", ec="r",lw = 1, label=r'Istogramma di $Q$')

    plt.xlim(xlims)
    x=np.linspace(*xlims, 1000)
    plt.plot(x, norm.pdf(x, 0, np.sqrt(Beta)),
             color='g',label=r'PDF attesa')
    plt.xlabel(r'$Q$')
    plt.ylabel(r'$P(Q)$')
    plt.savefig(rf"{path}/DistrQ_Nt={Nt_arr[Nt]}_beta={Beta}_eta={(Beta/Nt_arr[Nt]):.5f}_delta={delta:.5f}.png", bbox_inches='tight')
    plt.close()

def graphic_analysis(Nt, q, Beta):
    plt.plot(range(len(q)), q,
            label=f'{Nt}')
    plt.legend()
    bins=np.arange(q.min(), q.max()+2)
    bins=bins-0.5
    plt.hist(q, bins, density=True,
            histtype='step', fill=False,
            color='b', label=r'Istogramma di $Q$')
    xlims=[-15,15]
    plt.xlim(xlims)
    x=np.linspace(*xlims, 1000)
    plt.plot(x, norm.pdf(x, 0, np.sqrt(Beta)),
            color='g', label=r'PDF attesa')
    plt.xlabel(r'$Q$')
    plt.ylabel(r'$P(Q)$')

def connected_time_correlation(x: np.array, max_time, normalized=False) -> np.array:
    if max_time > len(x) - 1:
        raise IndexError
    x_mean = x.mean()
    C = np.array(
        [(x * x).mean() - x_mean**2] +
        [(x[:-k] * x[k:]).mean() - x_mean**2 for k in range(1, max_time)])
    if normalized:
        if C[0] != 0:
            return C / C[0]
        else:
            print("C[0] is zero, returning unormalized correlations")
    return C

def exponential(x, a, tau):
    return a * np.exp(-x/tau)

def correlation_time(x, label=None):
    C_LIMIT = 0.005
    T_LIMIT = 1000
    max_time = int(len(x)/20)
    C = connected_time_correlation(x, max_time, normalized=True)
    stop = np.argmin(C > C_LIMIT)
    if stop == 0:
        stop = T_LIMIT
    elif stop == 1:
        stop = 3
    C_fit = C[:stop]
    opt, cov = curve_fit(exponential, np.arange(stop), C_fit, p0=(1, 1))
    # Plots
    # plt.plot(range(max_time), C, color='red', label=label)
    # t_plot = np.linspace(0, stop, num=100)
    # plt.plot(t_plot, exponential(t_plot, *opt))
    # plt.legend()
    # plt.show()
    print(opt[1])
    return opt[1]
