import sys
import numpy as np
import pandas as pd
import argparse
import json
from reaction_tools import get_number_of_reaction
import math

np.set_printoptions(precision=3)

parser = argparse.ArgumentParser()
parser.add_argument("--reac_json", default="reaction_energy.json",
                                   help="json for reading reaction energy and writing rate")

args = parser.parse_args()
reac_json = args.reac_json

debug = False

T = 700  # K
ptot = 100.0  # bar
kJtoeV = 1/98.415
kB = 8.617e-5  # eV/K
RT = 8.314*1.0e-3*T*kJtoeV  # J/K -> kJ/K * K -> eV

gas = {"H2O": 0, "H2": 1}
ads = {"OH": 0, "O": 1, "OOH": 2, "vac": 3}

# --------------------------
reactionfile = "oer.txt"
rxn_num = get_number_of_reaction(reactionfile)

# entropy in eV/K
# loss in each elementary reaction
deltaS    = np.zeros(rxn_num)
deltaS[0] = -1.98e-3  # N2
deltaS[1] = -1.35e-3  # H2

# zero-point energy (ZPE) in eV
# loss in each elementary reaction
deltaZPE    = np.zeros(rxn_num)
deltaZPE[0] = -0.145  # N2
deltaZPE[1] = -0.258  # H2

# thermal correction (translation + rotation) in eV
# loss in each elementary reaction
deltaTherm    = np.zeros(rxn_num)
deltaTherm[0] = -(3/2)*RT - RT  # N2
deltaTherm[1] = -(3/2)*RT - RT  # H2

# pressure correction i.e. deltaG = deltaG^0 + RT*ln(p/p0)
# loss in each elementary reaction
RTlnP = np.zeros(rxn_num)
#RTlnP[0] -= RT*np.log(p[gas["N2"]])

# reation energies (in eV) and equilibrium constant
df_reac  = pd.read_json(reac_json)
df_reac  = df_reac.set_index("unique_id")
num_data = len(df_reac)

for id in range(num_data):
    #if "rate" in df_reac.iloc[id].index:
    # already calculated
    #	pass
    if isinstance(df_reac.iloc[id].reaction_energy, list):
        unique_id = df_reac.iloc[id].name
        deltaE  = df_reac.iloc[id].reaction_energy
        deltaE  = np.array(deltaE)
        eta = np.max(deltaE)  # overpotential

        deltaH  = deltaE + deltaZPE + deltaTherm

        deltaG  = deltaH - T*deltaS
        deltaG += RTlnP

        K = np.exp(-deltaG/(kB*T))

        score = -eta  # smaller is better

        data = {"unique_id": unique_id, "reaction_energy": list(deltaE),
                "gibbs_energy": list(deltaG), "temperature": T, "total_pressure": ptot,
                "species": list(ads.keys()), "score": score}

        # add to json
        with open(reac_json, "r") as f:
            datum = json.load(f)
            for i in range(len(datum)):
                if datum[i]["unique_id"] == unique_id:
                    datum.pop(i)
                    break

            datum.append(data)
            with open(reac_json, "w") as file:
                json.dump(datum, file, indent=4)

    else:
        print("reaction energy not available")
