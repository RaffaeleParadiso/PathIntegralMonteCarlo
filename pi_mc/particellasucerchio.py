import matplotlib.pyplot as plt
import numpy as np
import func as fn

Nt = 1800 # numero passi in cui è discretizzata la circonferenza
N = 10 # numero cammini
a = (5.0)/Nt  # la spaziatura temporale
y = np.zeros(shape=(Nt,))
ym = np.zeros(shape=(Nt,))
yp = np.zeros(shape=(Nt,))
npp, nmm = fn.geometry(Nt)
term = 10000
salto = 10
D = (N-term)/salto
delta = a**0.5

for i in range(0, Nt):
    y[i] = y[i-1] + np.random.random()
    if i == Nt-1:
        y[Nt-1] = y[0]

for i in range(0, Nt):
    yp[i] = y[npp[i]] # E' il vettore dei valori di y precedenti a ogni sito del vettore y
    ym[i] = y[nmm[i]] # E' il vettore dei valori di y successivi a ogni sito del vettore y

avvol = np.zeros(shape=(500,))
for _ in range(0,500):
    for j in range(0, N):
        sd = fn.metropolis_locale(Nt,y,ym,yp, delta, a)
    avvol[_] = sd

print(len(avvol))
plt.figure()
plt.plot(np.arange(len(avvol)), avvol)
plt.show()
print(avvol)