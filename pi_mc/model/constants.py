import numpy as np

NT = 10000
BETA = 10
ETA = BETA/NT
DELTA_METRO=0.5
PATHS=1000000
TERM=10000
NT_ARRAY=np.arange(100, 450, 50)
ETA_ARRAY = BETA/(np.array(NT_ARRAY))
N_BINS=7
BIN_INIT=1
BIN_ARRAY=np.array([BIN_INIT*2**i for i in range(0, N_BINS)])
