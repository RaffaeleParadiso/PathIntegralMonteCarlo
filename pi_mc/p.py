import numpy as np

a = np.loadtxt("meas_out")
import statistics
print(statistics.mean((a[:,1]**2)))
