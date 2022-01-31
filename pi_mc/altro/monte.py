import numpy as np

Nt = 5
a = (2.0)/Nt
delta = a**0.5

q = np.random.uniform(0,1,Nt)
ym = np.zeros(shape=(Nt,))
yp = np.zeros(shape=(Nt,))
for i in range(0, Nt):
    y = np.copy(q)
    y[i] = np.random.uniform(q[i]-delta, q[i]+delta)
    y[Nt-1] = y[0]
    yp[i] = y[i-1]
    yp[0] = y[0]
    yp[Nt-1] = y[Nt-2]
    ym[0] = y[1]
    ym[Nt-1] = y[Nt-1]
    if i < Nt-1:
        ym[i] = y[i+1]

from func import geometry
ym=np.zeros(shape=(Nt,))
yp=np.zeros(shape=(Nt))
npp, nmm = geometry(Nt)
for i in range(0,Nt):
    ym[i]=y[npp[i]]
    yp[i]=y[nmm[i]]

print(y)
print(ym)
print(yp)