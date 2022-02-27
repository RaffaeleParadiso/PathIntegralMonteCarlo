import numpy as np

# temperature
BETA = 2

# different discretization
NT_ARRAY=np.arange(200, 700, 50)
ETA_ARRAY = BETA/(np.array(NT_ARRAY))
DELTA_L = np.sqrt(ETA_ARRAY)
# metropolis parameters
DELTA_METRO=0.5
PATHS=1000000
TERM=50000

# bootstrap parameters
N_BINS=2
BIN_INIT=1010
BIN_ARRAY=np.array([BIN_INIT*i for i in range(1, N_BINS)])
