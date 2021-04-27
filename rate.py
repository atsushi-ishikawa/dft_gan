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

# --------------------------
reactionfile = "nh3.txt"
rxn_num = get_number_of_reaction(reactionfile)

# pressure in Pa
p = {"N2": 1.0e5, "H2": 1.0e5, "NH3": 0.1e5}

# coverage
theta  = np.zeros(rxn_num)

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
theta[0] = p["N2"] / K[0]
theta[1] = p["N2"] / np.sqrt(K[1])
theta[2] = p["N2"] / np.sqrt(K[2])
theta[3] = p["N2"] / np.sqrt(K[3])
theta[4] = p["N2"] / np.sqrt(K[4])

sum = 0.0
for i in range(len(theta)-1):
	sum += theta[i]
theta[-1] = 1 - sum
	
Keq = 1.0
for i in range(rxn_num):
	Keq *= K[i]

gamma = (1/Keq)*(p["NH3"]**2/(p["N2"]*p["H2"]**3))
rate  = k*p["N2"]*theta[-1]**2*(1-gamma)

print("rate = %e" % rate)
data = {"unique_id": unique_id, "reaction_energy": list(deltaE), "status": "done", "rate": rate}
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
