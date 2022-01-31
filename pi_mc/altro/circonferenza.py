import numpy as np
import matplotlib.pyplot as plt
from numba import njit

@njit(cache=True)
def distanza(x,y):
    diff = y - x
    if (diff > 0.5):
        return diff - 1
    if (diff < -0.5):
        return 1 + diff
    else:
        return diff

@njit(cache=True)
def avvolgimento(Nt, y, g):
    result = 0.0
    for i in range(Nt):
        result += distanza(y[modulo(i+1, Nt)], y[i])
    return int(round(result))

@njit(cache=True)
def geometry(Nt):
    npp = [i+1 for i in range(0,Nt)]
    nmm = [i-1 for i in range(0,Nt)]
    npp[Nt-1] = 0
    nmm[0] = Nt-1
    return (npp, nmm)

def modulo(b,c):
    return ((b % c) + c) % c


@njit(cache=True)
def give(Nt, delta):
    y = np.zeros(Nt)
    ym = np.zeros(shape=(Nt,))
    yp = np.zeros(shape=(Nt,))
    npp, nmm = geometry(Nt)
    for i in range(0, Nt):
        y[i] = np.random.uniform(y[i]-delta, y[i]+delta)    
        if(y[i] > 1): y[i] = y[i] -1.0
        if(y[i] < 0): y[i] = y[i] + 1.0 
    y[Nt-1] = y[0]
    yp[i]=y[npp[i]]
    ym[i]=y[nmm[i]]
    return y, yp, ym

Nt = 500
a = 2.0/Nt
delta = a**0.5
y, yp, ym = give(Nt, delta)
li = [x*a for x in range(0,Nt)]
Q = avvolgimento(Nt,y, yp)
print(Q)
plt.figure()
plt.plot(y, li)
plt.show()