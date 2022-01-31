import numpy as np
from numba import njit

@njit()
def distanza(x, y):
    if (abs(x-y) <= 0.5):
        d = x-y
    if ((x-y) < -0.5):
        d = x-y+1.0
    if ((x-y) > 0.5):
        d = x-y-1.0
    return d

@njit(fastmath=True, cache=True)
def diff_azione(i, y, g, f, yp1):
    d1 = distanza(g[i], yp1)
    d2 = distanza(g[i], y[i])
    d3 = distanza(yp1, f[i])
    d4 = distanza(y[i], f[i])
    return (d1*d1)+(d3*d3)-(d2*d2)-(d4*d4)

@njit(fastmath=True, cache=True)
def geometry(Nt):
    npp = [i+1 for i in range(0, Nt)]
    nmm = [i-1 for i in range(0, Nt)]
    npp[Nt-1] = 0
    nmm[0] = Nt-1
    return (npp, nmm)

@njit(fastmath=True, cache=True)
def avvolgimento(Nt, y, g):
    sum = 0
    for i in range(0, Nt):
        sum +=(distanza(g[i], y[i]))
    return sum

@njit(fastmath=True, cache=True)
def metropolis(y, Nt, a): #ymm va avanti e ypp va indietro
    y_new=y.copy()
    delta = np.sqrt(a)
    ypp, ymm = np.zeros(Nt), np.zeros(Nt)
    npp, nmm = geometry(Nt)
    for i in range(Nt):
        r = np.random.uniform(-delta, delta)
        yprova = y_new[i]+r
        if yprova > 1:
            yprova -= 1
        if yprova < 0:
            yprova += 1
        ypp[i] = y_new[npp[i]]
        ymm[i] = y_new[nmm[i]]
        s = diff_azione(i, y_new, ymm, ypp, yprova)
        if s <= 0:
            y_new[i] = yprova
        else:
            if np.random.rand() < np.exp(-s/(2.0*a)):
                y_new[i] = yprova
                y_new[Nt-1]=y_new[0]
    for i in range(Nt):
        ypp[i]=y_new[npp[i]]
        ymm[i]=y_new[nmm[i]]      
    Q = avvolgimento(Nt, y_new, ymm)
    return y_new, Q, ypp, ymm

@njit(fastmath=True, parallel=True)
def cammino_piano(Nt, a, cammini):
    q = []
    y = np.zeros(Nt)
    for i in range(Nt):
        y[i] = np.random.rand()
    for cam in range(cammini):
        for _ in range(10):
            y_new, Q, ypp, ymm = metropolis(y, Nt, a)
            y=y_new
        if cam > 10000:
            Q=avvolgimento(Nt, y, ymm)
            q.append(Q)
    return q, y, ypp, ymm

