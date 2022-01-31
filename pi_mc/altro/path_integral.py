import numpy as np
from numba import njit

def initialize_lattice(dim_latt, start=1):
    lattice_n = np.zeros((dim_latt))
    if start == 0:
        lattice_n = np.ones((dim_latt))
    if start == 1:
        for i in range(0, dim_latt):
            lattice_n[i]=1.0 - 2*np.random.random()
    return lattice_n

@njit(cache = True)
def geometry(dim_latt):
    npp = [i+1 for i in range(0,dim_latt)]
    nmm = [i-1 for i in range(0,dim_latt)]
    npp[dim_latt-1] = 0
    nmm[0] = dim_latt-1
    return (npp, nmm)

@njit(cache = True)
def metropolis(eta, d_metro, lattice_dim, lattice_n):
    c1 = 1./eta
    c2 = (1./eta + eta/2.)
    (npp, nmm) = geometry(lattice_dim)
    for i in range(lattice_dim):
        ip = npp[i]
        im = nmm[i]
        force = lattice_n[ip]+lattice_n[im]
        phi = lattice_n[i]
        x_rand = np.random.rand()
        phi_prova = phi+((2*d_metro)*(0.5-np.random.rand()))              
        p_rat = ((c1*phi_prova*force)-(c1*phi*force))+((c2*phi**2)-(c2*(phi_prova**2)))
        x = np.log(np.random.rand())                   
        if x <= p_rat:
            lattice_n[i] = phi_prova
    return lattice_n

def misure(i_term, d_metro, eta, measures, i_decorrel, lattice_n, lattice_dim=10):
    misure1 = []
    (npp, nmm) = geometry(lattice_dim)
    for i in range(i_term):
        m_term = metropolis(eta, d_metro , lattice_dim, lattice_n)
    for val in range(0, measures):
        for iter in range(0, i_decorrel):
            metr = metropolis(eta, d_metro , lattice_dim, m_term)
        o = 0.0
        for i in range(0, lattice_dim):
            o += +(metr[i]**2)           
            o = o/float(10)
        misure1.append(o)
    return misure1


eta = 0.1
d_metro = 0.5
measures = 1000000
i_decorrel = 10
i_term = 1

ret_ = initialize_lattice(10)

os1 = misure(i_term, d_metro, eta,measures, i_decorrel, ret_)
# np.savetxt("obs1.txt", os1)
import statistics
print(statistics.mean(os1))
