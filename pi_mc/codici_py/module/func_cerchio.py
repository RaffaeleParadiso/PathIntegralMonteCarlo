import numpy as np
import matplotlib.pyplot as plt
from numba import njit
# import constants as const

# Nt_array=const.NT_ARRAY
# cammini=const.CAMMINI
# term=const.TERM
cammini = 10000
term = 100

def observable(q):
    q2=(np.array(q))
    print(q2)
    var_naive=np.var(q2)
    print(var_naive)
    tau=0
    qki=np.zeros(len(q))
    for k in range(len(q)):
        qki[k]=q[0]*q[k]
    tau=0.5+(np.mean(qki)-np.mean(q)**2)/(var_naive)
    return tau

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
        r = np.random.normal(0, delta)
        yprova = y_new[i]+r
        if yprova > 1:
            yprova -= 1
        if yprova < 0:
            yprova += 1
        ypp[i] = y_new[npp[i]]  # come fa a sapere quello dopo?
        ymm[i] = y_new[nmm[i]]  # come fa a sapere quello prima per i = 0?
        s = diff_azione(i, y_new, ymm, ypp, yprova)
        if s <= 0:
            y_new[i] = yprova
        else:
            if np.random.rand() < np.exp(-s/(2.0*a)):
                y_new[i] = yprova
                y_new[Nt-1] = y_new[0]
    for i in range(Nt):
        ypp[i] = y_new[npp[i]]
        ymm[i] = y_new[nmm[i]]
    Q = avvolgimento(Nt, y_new, ypp)
    return y_new, Q, ypp, ymm

@njit(fastmath=True, cache=True)
def Tailor(y, epsilon, Nt, a):  # rif. all'articolo pag 7 algoritmo Tailor
    y0 = y[0]+0.5
    delta=np.sqrt(a)
    ypp, ymm = np.zeros(Nt), np.zeros(Nt)
    npp, nmm = geometry(Nt)
    if y0 >= 1.0:
        y0 = y0-1.0
    for i in range(1, Nt):
        dist = abs(distanza(y[i], y0))
        if dist <= epsilon:
            iend = i
            continue  # trovo il primo iend tale che l'if statement sopra sia soddisfatto()
    yprova=2*y0-y[iend] #valuto un certo yprova, ne calcolo la differenza di azione
    dS=distanza(y[iend+1], y[iend])**2-distanza(y[iend+1], yprova) #ora eseguo un test Metropolis
    if dS <= 0: 
        cambio=True
    else:
        if np.random.random()<=np.exp(-dS/(2*a)):  cambio =True
        else: cambio = False
    if cambio == True:
        for i in range(1, iend):
            y[i]=2.0*y0-y[i]
            if y[i]<0: y[i]+=1
            if y[i]>=1: y[i]-=1
            ypp[i]=y[npp[i]]
            ymm[i]=y[nmm[i]]
    return y, ypp, ymm

@njit(fastmath=True, parallel=True)
def cammino_piano(Nt):
    a = 2./Nt
    q = []
    y = np.zeros(Nt)
    for i in range(Nt):
        y[i] = np.random.rand()
    for cam in range(cammini):
        for _ in range(10):
            y_new, Q, ypp, ymm = metropolis(y, Nt, a)
            y = y_new
        if cam > term:
            if cam % 10 == 0:
                Q = avvolgimento(Nt, y, ymm)
                q.append(Q)
    return q, y, ypp, ymm

@njit(fastmath=True, cache=True)
def Tailor_exe(Nt):
    a=2/Nt
    epsilon=0.2*a
    q_list = []
    y=np.array([np.random.rand() for i in range(Nt)])
    for cam in range(cammini):
        if cam%100==0:
            print('sono al cammino', cam)
        y_tailor, ymm, y=cam_for(y, Nt, a, epsilon)
        if cam > term:
            if cam % 10 == 0:
                q_list.append(avvolgimento(Nt, y_tailor, ymm))
        y=y_tailor
    return q_list

@njit(fastmath=True, cache=True)
def cam_for(y, Nt, a, epsilon):
    for _ in range(1):
            y_metro, _, _, _ = metropolis(y, Nt, a)
            y = y_metro
    y_tailor, ypp, ymm =Tailor(y_metro, epsilon, Nt, a)
    return y_tailor, ymm, y


    
    
    


    
