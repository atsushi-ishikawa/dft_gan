import numpy as np

nrxn = 2

p = {"N2": 1.0, "H2": 1.0, "NH3": 0.1}

theta  = np.zeros(nrxn)
deltaE = np.zeros(nrxn)

deltaG = deltaE

K = np.exp(deltaG)

# Bronsted-Evans-Polanyi
alpha = 1.0
beta  = 1.0

Ea = alpha*deltaE + beta
kJtoeV = 1/98.415
RT = 8.314*300*1.0e-3*kJtoeV  # J/K * K
k = np.exp(-Ea/RT)

theta[0] = p["N2"] / K[0]
theta[1] = p["N2"] / np.sqrt(K[1])

Keq = K[0]*K[1]

gamma = (1/Keq)*(p["NH3"]**2/(p["N2"]*p["H2"]**3))
rate = k[0]*p["N2"]*theta[-1]**2*(1-gamma)

print("rate = %e" % rate)