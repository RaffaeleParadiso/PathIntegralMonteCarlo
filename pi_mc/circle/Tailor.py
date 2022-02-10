import numpy as np
import matplotlib.pyplot as plt
import module.func_cerchio as fnc
from numba import njit
from scipy.stats import norm

Nt=100000
beta=10
a=beta/Nt
epsilon=0.02*a
delta=np.sqrt(a)
y=np.array([np.random.normal(0, delta) for i in range(Nt)])
p_cut=0.06
term=20000
cammini=100000

@njit(fastmath=True, cache=True)
def Tailor(y, Nt):
    npp, nmm= fnc.geometry(Nt)
    q_list=[]
    for t in range(cammini):
        if t%1000==0: print(f'gno {t}')
        ypp, ymm=np.zeros(Nt), np.zeros(Nt)
        if np.random.rand()<p_cut:
            y0=(y[0]+0.5)%1
            y_new=y.copy()
            for i in range(Nt):
                if abs(fnc.distanza(y[i],y0))<=epsilon: iend=i
                continue
            yprova=(2*y0-y[iend])%1
            dS=fnc.distanza(yprova, y[iend-1])**2-fnc.distanza(y[iend], y[iend-1])**2
            if dS > 1 or np.random.rand() < np.exp(-dS/(2*a)):
                for m in range (iend, Nt):
                    y_new[m]=(2*y0-y[m])%1
                    ypp[m]=y_new[npp[m]]
                    ymm[m]=y_new[nmm[m]]
            y=y_new
            y[Nt-1] = y[0]
        for h in range(Nt):
            r=np.random.uniform(-delta, delta)
            y_old=y[h]
            y_bef=y[nmm[h]]
            y_aft=y[npp[h]]
            y_new1=(y_old+r)%1
            dS=fnc.distanza(y_aft, y_new1)**2+fnc.distanza(y_new1, y_bef)**2-fnc.distanza(y_aft, y_old)**2-fnc.distanza(y_old, y_bef)**2
            acceptance=np.exp(-dS/(2*a))
            if ((acceptance>1) or (np.random.rand()<acceptance)):
                y[h]=y_new1
            y[Nt-1] = y[0]
            ypp[h]=y[npp[h]]
            ymm[h]=y[nmm[h]]
        if t>term:
            q_list.append(fnc.avvolgimento(Nt,y, ymm))
    return q_list

q_list=Tailor(y, Nt)
q=np.array(q_list)
plt.plot(range(len(q)), q, label=f'{Nt}')
plt.legend()
plt.show()

bins=np.arange(q.min(), q.max()+2)
bins=bins-0.5
xlims=[-15,15]
x=np.linspace(*xlims, 1000)

plt.figure()
plt.hist(q, bins, density=True, histtype='step', fill=False, color='g', label=f'{Nt}')
plt.legend()
plt.xlim(xlims)
plt.plot(x, norm.pdf(x, 0, np.sqrt(beta)), color='r')
plt.show()

            

            


