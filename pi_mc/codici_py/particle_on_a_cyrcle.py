from dataclasses import field
import numpy as np
import matplotlib.pyplot as plt

#parametri
delta=0.5
eta=0.1
nlatt=30
measure=100000
idecorrel=10

def initialize_lattice(iflag):
    field=np.zeros(nlatt)
    if iflag == 1: field=0
    if iflag==1: 
        for i in range(nlatt): field[i]=1-2*np.random.rand()
    return field

def distance(x, y):
    if(abs(x - y) <= 0.5):  d =  x - y
    if((x - y) < -0.5): d = x - y + 1.0
    if((x - y)> 0.5): d = x - y - 1.0
    return d

def geometry():
    npp =[i+1 for i in range(0, nlatt)]
    nmm=[i-1 for i in range(0, nlatt)]
    nmm[nlatt-1]=0
    npp[0]=nlatt-1
    return npp, nmm

def avvolgimento(f, g):
    sum=0
    for i in range(0, nlatt):
        sum+=distance(f[i], g[i])
    return int(sum)


