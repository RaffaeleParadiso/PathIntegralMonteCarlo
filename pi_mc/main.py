import matplotlib.pyplot as plt
import numpy as np
import module.func as fn
import module.constants as cs

Nt=1000
cammini = 10000
term = 100
a=2./Nt
epsilon = 0.2*a

q, y_new = fn.cammino_piano(Nt, a, cammini, term)
plt.figure()
plt.plot(range(0, len(q)),q)
plt.show()

q = fn.cammino_Tailor(Nt, cammini, term, a, epsilon)
plt.figure()
plt.plot(range(0, len(q)),q)
plt.show()