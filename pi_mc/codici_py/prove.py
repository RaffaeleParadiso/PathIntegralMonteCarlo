import numpy as np
import matplotlib.pyplot as plt
import module.func_cerchio as fnc

Nt = 20
a = 2./Nt
cammini = 1
epsilon = 0.2*a
term=100
metro_each_tailor_inversion = 10
noTailor=False

# if noTailor:
#     q_no_tailor, _, _, _ = fnc.cammino_piano(Nt)
#     plt.figure()
#     plt.plot(range(len(q_no_tailor)), q_no_tailor)
#     plt.show()

# q_tailor=fnc.Tailor_exe(Nt)
# plt.figure()
# plt.plot(range(len(q_tailor)), q_tailor)
# plt.show()

# plt.show()
# tau=fnc.observable(q_no_tailor)
# print(tau)


q_list = []
plt.figure()
y=np.array([np.random.rand() for i in range(Nt)])
for cam in range(2):
    y_tailor, ymm, y=fnc.cam_for(y, Nt, a, epsilon)
    q_list.append(fnc.avvolgimento(Nt, y_tailor, ymm))
    y=y_tailor
    y0 = y[0]+0.5
    delta=np.sqrt(a)
    ypp, ymm = np.zeros(Nt), np.zeros(Nt)
    npp, nmm = fnc.geometry(Nt)
    if y0 >= 1.0:
        y0 = y0-1.0
    for i in range(1, Nt):
        dist = abs(fnc.distanza(y[i], y0))
        if dist <= epsilon:
            iend = i
            continue  # trovo il primo iend tale che l'if statement sopra sia soddisfatto()
    yprova=2*y0-y[iend] #valuto un certo yprova, ne calcolo la differenza di azione
    dS=fnc.distanza(y[iend+1], y[iend])**2-fnc.distanza(y[iend+1], yprova) #ora eseguo un test Metropolis
    if dS <= 0: 
        cambio=True
    else:
        if np.random.random()<=np.exp(-dS/(2*a)):  cambio =True
        else: cambio = False
    if cambio == True:
        for i in range(1, iend):
            y[i]=2.0*y0-y[i]
            if y[i]<0: y[i]+=1
            if y[i]>=1: y[i]-=1
            ypp[i]=y[npp[i]]
            ymm[i]=y[nmm[i]]
        print(i)


        plt.plot(range(len(y)), y)
plt.show()