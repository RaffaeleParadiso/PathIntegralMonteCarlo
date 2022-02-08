import matplotlib.pyplot as plt
import numpy as np
import module.func as fn
import module.constants as cs
from scipy.stats import norm

Nt=1000
cammini = 1000000
term = 10000
a=1/Nt
epsilon = 0.2*a

Q, y_new = fn.cammino_piano(Nt, a, cammini, term)
# plt.figure()
# plt.plot(range(0, len(q)),q)
# plt.show()


# Q, _, _, _ = fn.cammino_piano(Nt)
Q = np.array(Q)
bins = np.arange(Q.min(), Q.max()+2)
bins = bins - 0.5
plt.hist(Q, bins, density=True, histtype='step', fill=False, color='b', label=r'Istogramma di $Q$')
xlims = [-15, 15]
plt.xlim(xlims)
x = np.linspace(*xlims,1000)
# x = np.linspace(Q.min() - 2, Q.max() + 2, 1000)
plt.plot(x, norm.pdf(x, 0, np.sqrt(10)), color='g', label=r'PDF attesa')
plt.xlabel(r'$Q$')
plt.ylabel(r'$P(Q)$')
plt.show()


# q = fn.cammino_Tailor(Nt, cammini, term, a, epsilon)
plt.figure()
plt.plot(range(0, len(Q)),Q)
plt.show()