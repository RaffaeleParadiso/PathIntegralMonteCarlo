import matplotlib.pyplot as plt
import numpy as np
import func  as fn

Nt = 1000  # numero passi in cui è discretizzata la circonferenza
N = 10  # numero cammini
a = (5.0)/Nt  # la spaziatura temporale
delta = a**0.5
monte = 10000
term = 1000

avvol = np.zeros(shape=(monte,))
for _ in range(0, monte):
    if _ % 1000 == 0:
        print(_)
    for __ in range(0, N):
        y, ym, yp = fn.give(Nt)
        sd = fn.metropolis_locale(Nt, a, y, ym, yp)
    if _ > term:
        avvol[_] = sd


plt.figure()
plt.plot(np.arange(len(avvol)), avvol)
plt.show()


# plt.figure()
# li = [x*a for x in range(0,Nt)]
# plt.axhline(y = 0, color = 'b', label = 'axvline - full height')
# plt.axhline(y = 1, color = 'b', label = 'axvline - full height')
# plt.plot(li, y)
# plt.show()