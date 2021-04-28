import sys
import numpy as np
import pandas as pd
import argparse
import json
from reaction_tools import get_number_of_reaction

parser = argparse.ArgumentParser()
parser.add_argument("--id")
parser.add_argument("--reac_json", default="reaction_energy.json", help="json for reading rxn. energy and writing rate")

args = parser.parse_args()
unique_id = args.id
reac_json = args.reac_json

gas = {"N2": 0, "H2":1, "NH3": 2}
ads = {"N" : 0, "H":1, "NH": 2, "NH2": 3, "NH3": 4, "vac": 5}

# --------------------------
reactionfile = "nh3.txt"
rxn_num = get_number_of_reaction(reactionfile)

# pressure in Pa
p = np.zeros(len(gas))
conversion = 0.1
p[gas["N2"]]  = 1.0e5
p[gas["H2"]]  = p[gas["N2"]]*3.0
p[gas["NH3"]] = p[gas["N2"]]*conversion

# coverage
theta  = np.zeros(len(ads))

# reation energies and equilibrium constant
df_reac = pd.read_json(reac_json)
df_reac = df_reac.set_index("unique_id")
if unique_id in df_reac.index:
	try:
		deltaE = df_reac.loc[unique_id].reaction_energy
	except:
		print("reaction energy not found on %s" % reac_json)
		sys.exit(1)

deltaE = np.array(deltaE)
deltaG = deltaE
K = np.exp(deltaG)

# rate-determining step
rds = 0

# activation energy
# Bronsted-Evans-Polanyi
alpha = 1.5
beta  = 2.0
tmp = alpha*deltaE + beta
Ea  = tmp[rds]

kJtoeV = 1/98.415
RT = 8.314*300*1.0e-3*kJtoeV  # J/K * K
k = np.exp(-Ea/RT)

# coverage
tmp = 1 + np.sqrt(K[1]*p[gas["H2"]]) \
		+ p[gas["NH3"]]/(np.sqrt(K[1]*p[gas["H2"]])*K[4]+K[5]) \
		+ p[gas["NH3"]]/(K[1]*p[gas["H2"]]*K[3]*K[4]*K[5]) \
		+ p[gas["NH3"]]/(K[1]**(3/2)*p[gas["H2"]]**(3/2)*K[2]*K[3]*K[4]*K[5]) \
		+ p[gas["NH3"]]/K[5]

theta[ads["vac"]] = 1/tmp
theta[ads["H"]]   = np.sqrt(K[1]*p[gas["H2"]])*theta[ads["vac"]]
theta[ads["NH3"]] = (p[gas["NH3"]]/K[5])*theta[ads["vac"]]
theta[ads["NH2"]] = (p[gas["NH3"]]/(np.sqrt(K[1]*p[gas["H2"]])*K[4]*K[5]))*theta[ads["vac"]]
theta[ads["NH"]]  = (p[gas["NH3"]]/(K[1]*p[gas["H2"]]*K[3]*K[4]*K[5]))*theta[ads["vac"]]
theta[ads["N"]]   = (p[gas["NH3"]]/(K[1]**(3/2)*p[gas["H2"]]**(3/2)*K[2]*K[3]*K[4]*K[5]))*theta[ads["vac"]]

Keq     = K[0]*K[1]**3*K[2]**2*K[3]**2*K[4]**2*K[5]**2
gamma   = (1/Keq)*(p[gas["NH3"]]**2/(p[gas["N2"]]*p[gas["H2"]]**3))
rate    = k*p[gas["N2"]]*theta[ads["vac"]]**2*(1-gamma)
lograte = np.log10(rate)

data = {"unique_id": unique_id, "reaction_energy": list(deltaE), "status": "done", "rate": lograte}
# add to json
with open(reac_json, "r") as f:
	datum = json.load(f)
	for i in range(len(datum)):
		if datum[i]["unique_id"] == unique_id:
			datum.pop(i)
			break
	datum.append(data)
with open(reac_json, "w") as f:
	json.dump(datum, f, indent=4)
