import numpy as np
from numba import njit

@njit(cache=True)
def distanza(x,y): # x = y[i+1], y = y[i]
	if(abs(x - y) <= 0.5): d =  x - y
	if((x - y) < -0.5): d = x - y + 1.0
	if((x - y)> 0.5): d = x - y - 1.0
	return d

@njit(cache=True)
def avvolgimento(Nt, g, y):
    sum = 0.0
    for i in range(0,Nt):
        sum = sum + distanza(g[i],y[i])
    avv = sum
    return avv

@njit(cache=True)
def diff_azione(i, y, g, f, yp1):
    d1 = distanza(g[i],yp1)
    d2 = distanza(g[i],y[i])
    d3 = distanza(yp1,f[i])
    d4 = distanza(y[i],f[i])
    return (d1*d1)+(d3*d3)-(d2*d2)-(d4*d4)

@njit(cache=True)
def geometry(Nt):
    npp = [i+1 for i in range(0,Nt)]
    nmm = [i-1 for i in range(0,Nt)]
    npp[Nt-1] = 0
    nmm[0] = Nt-1
    return (npp, nmm)

@njit(cache=True)
def give(Nt):
    q = np.random.uniform(0,1,Nt)
    ym = np.zeros(shape=(Nt,))
    yp = np.zeros(shape=(Nt,))
    npp, nmm = geometry(Nt)
    for i in range(0, Nt):
        y = np.copy(q)
        y[i] = np.random.uniform(0,1)      
        y[Nt-1] = y[0]
        yp[i]=y[npp[i]]
        ym[i]=y[nmm[i]]
    return y, yp, ym

@njit(cache=True)
def metropolis_locale(Nt, a, y, ym, yp):
    for i in range(0, Nt):
        yp1 = 0.0
        npp, nmm = geometry(Nt)
        r = np.random.random()
        yp1 = (y[i] + (1-2*r)*0.5)
        if(yp1 > 1): yp1 = yp1 -1.0
        if(yp1 < 0): yp1 = yp1 + 1.0
        S = diff_azione(i,y,yp,ym,yp1)
        if S<=0: y[i] = yp1
        else:
            if (np.random.random() < np.exp(-S/(2.0*a))): y[i] = yp1
        y[Nt-1] = y[0]
    avv = avvolgimento(Nt, yp, y)
    return avv