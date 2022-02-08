import matplotlib.pyplot as plt
from numba import njit
import numpy as np
from scipy.stats import norm

@njit(cache=True)
def distance(y, x):
    diff = y - x
    if (diff > 0.5):  return diff - 1
    if (diff < -0.5): return 1 + diff
    else:             return diff

@njit(cache=True)
def modul(a, b):
    return ((a % b) + b) % b

@njit(cache=True)
def charge(Nt,y):
    result = 0.
    for i in range(0, Nt):
        result += distance(y[modul(i+1, Nt)], y[i])
    return round(result)

@njit(cache=True)
def avvolgimento(Nt, y, g):
    sum = 0.0
    for i in range(0, Nt):
        sum +=(distance(g[i], y[i]))
    return round(sum)

@njit(cache=True)
def geometry(Nt):
    npp = [i+1 for i in range(0, Nt)]
    nmm = [i-1 for i in range(0, Nt)]
    npp[Nt-1] = 0
    nmm[0] = Nt-1
    return (npp, nmm)

Nt = 1000
beta = .5
a = beta/Nt
delta = a**(0.5)
cammini = 100000
npp, nmm = geometry(Nt)
# delta = 0.5

y = np.zeros(shape=(Nt,))
for i in range(0, Nt-1):
    y[i] = np.random.random()
y[Nt-1] = y[0]

def met():
    Q = []
    for i in range(0, cammini):
        ypp, ymm = np.zeros(Nt), np.zeros(Nt)
        for i in range(0, Nt):
            ypp[i] = y[npp[i]]
            ymm[i] = y[nmm[i]]
        for i in range(10):
            for _ in range(0, Nt):
                y_old = y[_]
                y_before = ymm[_]
                y_after = ypp[_]
                x = np.random.uniform()
                y_new = y_old +(delta*(2*x-1))
                if y_new>1: y_new = y_new-1
                if y_new<0: y_new = y_new+1
                DeltaSE = (1./(2*a)) * distance(y_after, y_new)**2 + distance(y_new, y_before)**2 - distance(y_after, y_old)**2 - distance(y_old, y_before)**2
                if (DeltaSE < 0) or (np.random.uniform() < np.exp(-DeltaSE)):
                    y[_] = y_new
                y[Nt-1] = y[0]
        Q.append(charge(Nt, y)) 
    return Q

Q = met()
Q = np.array(Q)
bins = np.arange(Q.min(), Q.max()+2)
bins = bins - 0.5
plt.hist(Q, bins, density=True, histtype='step', fill=False, color='b', label=r'Istogramma di $Q$')
xlims = [-15, 15]
plt.xlim(xlims)
x = np.linspace(*xlims,1000)
# x = np.linspace(Q.min() - 2, Q.max() + 2, 1000)
plt.plot(x, norm.pdf(x, 0, beta), color='g', label=r'PDF attesa')
plt.xlabel(r'$Q$')
plt.ylabel(r'$P(Q)$')
plt.show()
plt.figure()
plt.plot(range(len(Q)), Q)
plt.show()
