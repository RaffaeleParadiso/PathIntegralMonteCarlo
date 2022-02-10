from re import A
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, float64
from scipy.stats import norm

### costanti
cammini = 10000
term = 100
Nt=100
beta=10

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
def avvolgimento(Nt, y, g):
    sum = 0
    for i in range(0, Nt):
        sum += (distanza(g[i], y[i]))
    return round(sum)

@njit(fastmath=True, cache=True)
def geometry(Nt):
    npp = [i+1 for i in range(0, Nt)]
    nmm = [i-1 for i in range(0, Nt)]
    npp[Nt-1] = 0
    nmm[0] = Nt-1
    return (npp, nmm)

@njit(fastmath=True, cache=True)
def metropolis(y, Nt, a):  # ymm va avanti e ypp va indietro
    y_new = y.copy()
    delta = np.sqrt(a)
    ypp, ymm = np.zeros(Nt), np.zeros(Nt)
    npp, nmm = geometry(Nt)
    for i in range(Nt):
        r = np.random.uniform(-delta, delta)
        yprova = (y_new[i]+r)%1
        # if yprova >= 1:
        #     yprova -= 1
        # if yprova < 0:
        #     yprova += 1
        ypp[i] = y_new[npp[i]]  # come fa a sapere quello dopo?
        ymm[i] = y_new[nmm[i]]  # come fa a sapere quello prima per i = 0?
        s = diff_azione(i, y_new, ypp, ymm, yprova)
        if s <= 0:
            y_new[i] = yprova
            y_new[Nt-1] = y_new[0]
        else:
            if np.random.normal(0, np.sqrt(a)) <= np.exp(-s/(2.0*a)):
                y_new[i] = yprova
                y_new[Nt-1] = y_new[0]
    for i in range(Nt):
        ypp[i] = y_new[npp[i]]
        ymm[i] = y_new[nmm[i]]
    Q = avvolgimento(Nt, y_new, ypp)
    return y_new, Q, ypp, ymm

@njit(fastmath=True, parallel=True)
def cammino_piano(Nt):
    a = 2./Nt
    q = []
    y = np.zeros(Nt)
    y = np.array([np.random.normal(0, np.sqrt(a)) for i in range(Nt)])
    for cam in range(cammini):
        if cam % 1000 == 0:
            print('sono al cammino metro', cam)
        for _ in range(10):
            y_new, Q, ypp, ymm = metropolis(y, Nt, a)
            y = y_new
        if cam > term:
            Q = avvolgimento(Nt, y, ypp)
            q.append(Q)
    return q

@njit(fastmath=True, cache=True)
def Tailor(y, Nt):

    a=beta/Nt
    delta=np.sqrt(a)
    p_cut=0.06
    epsilon=0.02*a 

    npp, nmm= geometry(Nt)
    q_list=[]
    for t in range(cammini):
        if t%1000==0: print(f'gno {t}')
        ypp, ymm=np.zeros(Nt), np.zeros(Nt)
        if np.random.rand()<p_cut:
            y0=(y[0]+0.5)%1
            y_new=y.copy()
            for i in range(Nt):
                if abs(distanza(y[i],y0))<=epsilon: iend=i
                continue
            yprova=(2*y0-y[iend])%1
            dS=distanza(yprova, y[iend-1])**2-distanza(y[iend], y[iend-1])**2
            if dS >1 : cambio=True 
            else:
                if np.random.rand() < np.exp(-dS/(2*a)): cambio = True
                else: cambio = False
            if cambio == True:
                for m in range (iend, Nt):
                    y_new[m]=(2*y0-y[m])%1
                    ypp[m]=y_new[npp[m]]
                    ymm[m]=y_new[nmm[m]]
            y=y_new
        for h in range(Nt):
            rand=np.random.randint(0, Nt)
            r=np.random.uniform(-delta, delta)
            y_old=y[rand]
            y_bef=y[nmm[rand]]
            y_aft=y[npp[rand]]
            y_new1=(y_old+r)%1
            dS=distanza(y_aft, y_new1)**2+distanza(y_new1, y_bef)**2-distanza(y_aft, y_old)**2-distanza(y_old, y_bef)**2
            acceptance=np.exp(-dS/(2*a))
            if ((acceptance>1) or (np.random.rand()<acceptance)):
                y[rand]=y_new1
            ypp[h]=y[npp[h]]
            ymm[h]=y[nmm[h]]
        y[Nt-1]=y[0]
        if t>term:
            q_list.append(avvolgimento(Nt,y, ymm))
    return q_list

def graphic_analysis(Nt, q):
    
    a=beta/Nt
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

