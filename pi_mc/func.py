import numpy as np
from numba import njit


@njit(cache=True)
def distanza(x,y):
    if (abs(x-y) <= 0.5):
        d= x-y
    if ((x-y) < -0.5):
        d=x-y+1.0
    if ((x-y) > 0.5):
        d=x-y-1.0
    return d

@njit(cache=True)
def avvolgimento(Nt, y, g):
    sum = 0.0
    for i in range(0,Nt-1):
        sum = sum + distanza(g[i],y[i])
    avv = int(sum)
    return avv

@njit(cache=True)
def diff_azione(i, y, g, f, yp1):
    """
    // Passo nella funzione l'indice i su cui faccio il Metropolis, 
	// a è la spaziatura temporale, 
	// o è il parametro del potenziale
	// yp la y di prova
    // I vettori g e f sono i corrispondenti di yp e ym rispettivamente
	// Definisco le quattro distanze che mi servono a calcolare la differenza di azione
	// y è il vettore che definisce il cammino, g è il vettore costituito dai punti successivi al sito i-esimo
	// f è il vettore dei punti che precedono il sito i-esimo
	// yp1 è l'y di prova da utilizzare nel Metropolis
    """
    d1 = distanza(g[i],yp1)
    d2 = distanza(g[i],y[i])
    d3 = distanza(yp1,f[i])
    d4 = distanza(y[i],f[i])
    return (d1*d1)+(d3*d3)-(d2*d2)-(d4*d4)

def azione_cammino(Nt, a, g, yp1):
    """
    // a è la spaziatura reticolare, 
	// yp1[] è il vettore che nel main diventerà il cammino dove ad un certo i sostituisco yi con yi di prova e ne valuto l'azione totale
    // il vettore g[] è il vettore dei punti successivi
    """
    s = 0.0
    for i in range(0, Nt):
        d = distanza(g[i], yp1[i])
        s = s + (1.0/(2.0*a))*(d*d)
    return s

def func_auto(dim, k, y):
    c = 0.0
    for i in range(0, dim):
        c = c + y[i]*y[(i+k)%dim]
    return c/dim

@njit(cache=True)
def geometry(Nt):
    npp = [i+1 for i in range(0,Nt)]
    nmm = [i-1 for i in range(0,Nt)]
    npp[Nt-1] = 0
    nmm[0] = Nt-1
    return (npp, nmm)

@njit(cache=True)
def metropolis_locale(Nt,y, yp, ym, delta, a):
    y = np.zeros(shape=(Nt,))
    for i in range(0, Nt):
        y[i] = y[i-1] + np.random.random()
        if i == Nt-1:
            y[Nt-1] = y[0]
    for i in range(0, Nt):
        yp1 = 0.0
        npp, nmm = geometry(Nt)
        yp[i] = y[npp[i]]
        ym[i] = y[nmm[i]]
        r = -delta + np.random.uniform(0,2*delta)
        yp1 = y[i] + r
        if yp1 > 1: 
            yp1 = yp1 - 1.0
        if yp1 < 0:
            yp1 = yp1 + 1.0
        S = diff_azione(i,y,yp,ym,yp1)
        if S<=0:
            y[i] = yp1
        else:
            if (np.random.random() < np.exp(-S/(2.0*a))):
                y[i] = yp1
            else:
                y[i] = y[i]
        if i == Nt-1:
            y[Nt-1] = y[0]

    Q = avvolgimento(Nt,y,yp)
    return Q
