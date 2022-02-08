import numpy as np
from numba import njit

@njit()
def distanza(x, y):
    diff = y - x
    if (diff > 0.5):
        return diff - 1
    if (diff < -0.5):
        return 1 + diff
    else:
        return diff

@njit(fastmath=True, cache=True)
def diff_azione(i, y, g, f, yp1):
    d1 = distanza(g[i], yp1)
    d2 = distanza(g[i], y[i])
    d3 = distanza(yp1, f[i])
    d4 = distanza(y[i], f[i])
    return (d1*d1)+(d3*d3)-(d2*d2)-(d4*d4)

@njit(fastmath=True, cache=True)
def avvolgimento(Nt, y, g):
    sum = 0.0
    for i in range(0, Nt):
        sum +=(distanza(g[i], y[i]))
    return round(sum)

@njit(fastmath=True, cache=True)
def geometry(Nt):
    npp = [i+1 for i in range(0, Nt)]
    nmm = [i-1 for i in range(0, Nt)]
    npp[Nt-1] = 0
    nmm[0] = Nt-1
    return (npp, nmm)

@njit(fastmath=True, cache=True)
def metropolis(y, Nt, a): #ymm va avanti e ypp va indietro
    y_new=y.copy()
    delta = 0.5
    # delta = a**(0.5)
    ypp, ymm = np.zeros(Nt), np.zeros(Nt)
    npp, nmm = geometry(Nt)
    for i in range(Nt):
        # r = np.random.uniform(y[i]-delta,y[i]+delta)
        # yprova = r - round(r)
        r = np.random.random()
        yprova = (y[i] + (1-2*r)*delta)%1
        ypp[i] = y_new[npp[i]]  # come fa a sapere quello dopo?
        ymm[i] = y_new[nmm[i]]  # come fa a sapere quello prima per i = 0?
        s = diff_azione(i, y_new, ymm, ypp, yprova)
        if s <= 0:
            y_new[i] = yprova
            y_new[Nt-1]=y_new[0]
        else:
            if (np.random.uniform(0,1) < np.exp(-s/(2.0*a))):
                y_new[i] = yprova
                y_new[Nt-1]=y_new[0]
    for i in range(Nt):
        ypp[i]=y_new[npp[i]]
        ymm[i]=y_new[nmm[i]]      
    Q = avvolgimento(Nt, y_new, ypp)
    return y_new, Q, ymm

@njit(fastmath=True, parallel=True)
def cammino_piano(Nt, a, cammini, term):
    q = []
    y = np.zeros(Nt)
    for i in range(Nt):
        y[i] = np.random.rand()
    for cam in range(cammini):
        for _ in range(10):
            y_new, Q, ymm = metropolis(y, Nt, a)
            y=y_new
        if cam > term:
            if cam%100==0:
                Q=avvolgimento(Nt, y, ymm)
                q.append(Q)
    return q, y_new

@njit(fastmath=True, parallel=True)
def cammino_Tailor(Nt, cammini, term, a, epsilon):
    q = []
    y = np.zeros(Nt)
    for i in range(Nt):
        y[i] = np.random.rand()
    for cam in range(cammini):
        for _ in range(10):
            y_new, ymm = Tailor(y, epsilon, Nt, a)
            y=y_new
        if cam > term:
            if cam%100==0:
                Q=avvolgimento(Nt, y_new, ymm)
                q.append(Q)
    return q

@njit(fastmath=True, cache=True)
def Tailor(y, epsilon, Nt, a):  # rif. all'articolo pag 7 algoritmo Tailor
    ypp, ymm = np.zeros(Nt), np.zeros(Nt)
    npp, nmm = geometry(Nt)
    indice_random = np.random.randind(0, Nt)
    for i in range(0, Nt):
        dist = (y[i] - (y[indice_random]-0.5))%(0.5)
        if dist <= epsilon:
            iend = i
            continue
    yprova = 2.0*y[0]-y[iend]
    if yprova < 0: yprova += 1
    if yprova >=1.0: yprova -= 1
    dS = distanza(y[iend+1], yprova)**2-distanza(y[iend+1], y[iend])**2
    if dS <= 0: 
        y[iend] = yprova
        ypp[i] = y[npp[i]]  # come fa a sapere quello dopo?
        ymm[i] = y[nmm[i]] 
    else:
        if np.random.rand() < np.exp(-dS/(2.*a)): 
            y[iend] = yprova
            ypp[i] = y[npp[i]]  # come fa a sapere quello dopo?
            ymm[i] = y[nmm[i]] 
    return y, ymm
