from operator import truediv
import numpy as np

NLATT=10
ETA=0.1
D_METRO=0.5
MEASURE=100000
I_DECORREL=10
#For particle on a circle:
CAMMINI=1000000
TERM=10000
NT_ARRAY=np.arange(100, 450, 50)
TAILOR=True
BOOTSTRAP=False
n_bins=7
bin_init=1
BIN_ARRAY=np.array([bin_init*2**i for i in range(0, n_bins)])
