import numpy as np
from numba import njit
import module.constants as const

nlatt=const.NLATT
eta=const.ETA
d_metro=const.D_METRO
measure=300000#const.MEASURE
i_decorrel=const.I_DECORREL

@njit(cache=True, fastmath=True)
def geometry(nlatt):
    npp = [i+1 for i in range(0,nlatt)]
    nmm = [i-1 for i in range(0,nlatt)]
    npp[nlatt-1] = 0
    nmm[0] = nlatt-1
    return (npp, nmm)

@njit(cache=True, fastmath=True)
def initialize_lattice(iflag):
    field=np.zeros(nlatt)
    if iflag==0:
        for i in range(nlatt):
            field[i]=0
    if iflag==1:
        for i in range(nlatt):
            x=1-2*np.random.rand()
            field[i]=x
    return field

@njit(cache=True, parallel=True)
def update_metropolis(field):
    c1=1/eta
    c2=1/eta+eta/2
    npp, nmm=geometry(nlatt)

    for i in range(nlatt):
        ip=npp[i]
        im=nmm[i]
        force= field[ip]+field[im]
        phi=field[i]
        x_rand=np.random.rand()
        phi_prova=phi+2*d_metro*(0.5-x_rand)
        p_rat=c1*phi_prova*force-c2*phi_prova**2
        p_rat=p_rat-c1*phi*force+c2*phi**2
        x=np.log(x_rand)
        if x<=p_rat: field[i]=phi_prova
    return field

@njit(cache=True, parallel=True) #field=y adimensionale
def misure(field):
    obs1=0.
    obs2=0.
    npp, nmm= geometry(nlatt)
    for i in range(nlatt):
        ip=npp[i]
        obs1+=+field[i]**2
        obs2+=+(field[i]-field[ip])**2
    obs1=obs1/float(nlatt) #medie sui singoli path, media su y
    obs2=obs2/float(nlatt) #variabile delta y
    return obs1, obs2

#calcolo vero e proprio 
@njit(cache=True, fastmath=True)
def calcoloveroeproprio(field):
    obser1=[]
    obser2=[]
    for i in range(measure):
        print('misura', i)
        for _ in range(i_decorrel):
            field=update_metropolis(field)
        obs1, obs2=misure(field)
        obser1.append(obs1)
        obser2.append(obs2)
    return (obser1, obser2)

field=initialize_lattice(1)
obser1, _=calcoloveroeproprio(field)
print(np.mean(obser1))


