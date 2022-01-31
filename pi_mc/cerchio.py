import matplotlib.pyplot as plt
import numpy as np
import func as fn

Nt=1000
cammini = 100000
term = 10000

y=np.zeros((Nt))
a=2./Nt 

for _ in range(Nt):
    y[_]=np.random.rand()

q, y, _, _ = fn.cammino_piano(Nt, a, cammini)
plt.figure()
plt.plot(range(len(q)), q)
plt.show()