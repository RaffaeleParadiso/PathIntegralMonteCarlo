import matplotlib.pyplot as plt
import numpy as np

def distanza(x,y):
    if (abs(x-y) <= 0.5):
        d= x-y
    if ((x-y) < -0.5):
        d=x-y+1.0
    if ((x-y) > 0.5):
        d=x-y-1.0
    return d

def avvolgimento(Nt, y, g):
    sum = 0.0
    for i in range(0,Nt-1):
        sum = sum + distanza(g[i],y[i])
    avv = int(sum)
    return avv

def diff_azione(i, y, g, f, yp1):
    d1 = distanza(g[i],yp1)
    d2 = distanza(g[i],y[i])
    d3 = distanza(yp1,f[i])
    d4 = distanza(y[i],f[i])
    return (d1*d1)+(d3*d3)-(d2*d2)-(d4*d4)

def geometry(Nt):
    npp = [i+1 for i in range(0,Nt)]
    nmm = [i-1 for i in range(0,Nt)]
    npp[Nt-1] = 0
    nmm[0] = Nt-1
    return (npp, nmm)

def azione_cammino(Nt, g, yp1, a):
    s = 0.0 # azione totale
    for i in range(0, Nt):
        Di = distanza(g[i], yp1[i])
        s = s + (1.0/(2.0*a))*(Di**2)
    return s

def metropolis(Nt, y):
	(npp, nmm) = geometry(Nt)
	for i in range(0, Nt):
		y[i] = y[i] + np.random.uniform(-sqet, +sqet)
		a = distanza(y[i], y[i-1])
	return a

def run_metropolis(lattice_dim, ret_, i_decorrel, measures):
    for val in range(0, measures):
        for iter in range(0, i_decorrel):
            metr = metropolis(lattice_dim, ret_)
    return 


Nt = 50 # numero passi
N = 10 # numero cammini
eta = 0.04  # la spaziatura temporale
Nt_eta = Nt*eta # qualcosa fissato
sqet = np.sqrt(Nt_eta)
y = [np.random.random() for i in range(0,Nt)] # array cammino
ym = np.zeros(shape=(Nt,))
yp = np.zeros(shape=(Nt,))
y = np.array(y)

npp, nmm = geometry(Nt)

for i in range(0, Nt):
    yp[i] = y[npp[i]]
    ym[i] = y[nmm[i]]
