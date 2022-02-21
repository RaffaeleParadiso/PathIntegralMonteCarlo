import numpy as np

# temperature
BETA = 2

# discretization for the path on the circumference
NT = 10000
ETA = BETA/NT

# different discretization
NT_ARRAY=np.arange(200, 600, 50)
ETA_ARRAY = BETA/(np.array(NT_ARRAY))

# metropolis parameters
DELTA_METRO=0.5
PATHS=100000
TERM=10000

# bootstrap parameters
N_BINS=7
BIN_INIT=10
BIN_ARRAY=np.array([BIN_INIT*2**i for i in range(0, N_BINS)])
