import numpy as np
from numba import njit

def initialize_lattice(dim_latt, start=1):
    lattice_n = np.zeros((dim_latt))
    if start == 0:
        lattice_n = np.ones((dim_latt))
    if start == 1:
        for i in range(0, dim_latt):
            lattice_n[i]=np.random.random()
    return lattice_n

@njit(cache = True)
def geometry(dim_latt):
    npp = [i+1 for i in range(0,dim_latt)]
    nmm = [i-1 for i in range(0,dim_latt)]
    npp[dim_latt-1] = 0
    nmm[0] = dim_latt-1
    return (npp, nmm)

@njit(cache = True)
def metropolis(eta, delta_m, dim_latt, lattice_n):
    c1 = 1./eta
    c2 = (1./eta + eta/2.)
    (npp, nmm) = geometry(dim_latt)
    for i in range(dim_latt):
        ip_ = npp[i]
        im_ = nmm[i]
        force = lattice_n[ip_]+lattice_n[im_]
        phi = lattice_n[i]
        x_rand = np.random.random()
        phi_prova = phi + 2.*delta_m*(0.5-x_rand)                  
        p_rat = c1 * phi_prova * force - c2 * phi_prova**2
        p_rat = p_rat - c1 * phi * force + c2 * phi**2   
        x = np.log(x_rand)                   
        if x < p_rat: 
            lattice_n[i] = phi_prova
    return lattice_n


@njit(cache = True)
def run_metropolis(delta_m, eta, measures, i_decorrel, npp, lattice_n, lattice_dim=10):
    obs1 = 0.0
    obs2 = 0.0
    for val in range(0, measures):
        for iter in range(0, i_decorrel):
            metr = metropolis(eta, delta_m , lattice_dim, lattice_n)
        obs1 = obs1 + metr[iter]**2
        obs1 = (obs1/float(lattice_dim))
        obs2 = obs2 + metr[iter]-lattice_n[npp[iter]]**2       
        obs2 = (obs2/float(lattice_dim))
    return (obs1, obs2)
