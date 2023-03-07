import sys
import numpy as np
import pandas as pd
import argparse
import json
from reaction_tools import get_number_of_reaction
import math

np.set_printoptions(precision=3)

parser = argparse.ArgumentParser()
parser.add_argument("--reac_json", default="reaction_energy.json", help="json for reading reaction energy and writing rate")

args = parser.parse_args()
reac_json = args.reac_json

debug = False

T = 298.15  # K
kJtoeV = 1/98.415
kB = 8.617e-5  # eV/K

# --------------------------
reactionfile = "oer.txt"
rxn_num = get_number_of_reaction(reactionfile)

## OER
# 0) H2O + * -> OH* + H+ + e-
# 1) OH* -> O* + H+ + e-
# 2) O* + H2O -> OOH* + H+ + e-
# 3) OOH* -> * + O2 + H+ + e-
#  Using CHE, chemical potential of H+ + e- becomes that of 0.5*H2.
#  O2 Gibbs energy is replaced: O2 + 2H2 <-> 2H2O + 4.92
#
species = {"H2" : 0, "H2O": 1, "OHads": 2, "Oads": 3, "OOHads": 4, "O2": 5}
E_redox = {"OER": 0.758}

# entropy in eV/K
S = np.zeros(len(species))

S[species["H2"]]  = 0.41/T
S[species["H2O"]] = 0.67/T
S[species["O2"]]  = 0.32*2/T

# ZPE in eV
zpe = np.zeros(len(species))
zpe[species["H2"]]     = 0.27
zpe[species["H2O"]]    = 0.56
zpe[species["OHads"]]  = 0.36
zpe[species["Oads"]]   = 0.07
zpe[species["OOHads"]] = 0.40  # temporary 
zpe[species["O2"]]     = 0.05*2

# loss in each elementary reaction
deltaS    = np.zeros(rxn_num)
deltaS[0] = 0.5*S[species["H2"]] - S[species["H2O"]]
deltaS[1] = 0.5*S[species["H2"]]
deltaS[2] = 0.5*S[species["H2"]] - S[species["H2O"]]
#deltaS[3] = S[species["O2"]] + 0.5*S[species["H2"]]
deltaS[3] = 2.0*S[species["H2O"]] -1.5*S[species["H2"]]

# zero-point energy (ZPE) in eV
deltaZPE    = np.zeros(rxn_num)
deltaZPE[0] = zpe[species["OHads"]] + 0.5*zpe[species["H2"]] - zpe[species["H2O"]]
deltaZPE[1] = zpe[species["Oads"]] + 0.5*zpe[species["H2"]] - zpe[species["OHads"]]
deltaZPE[2] = zpe[species["OOHads"]] + 0.5*zpe[species["H2"]] - zpe[species["Oads"]] - zpe[species["H2O"]]
#deltaZPE[3] = zpe[species["O2"]] + 0.5*zpe[species["H2"]] - zpe[species["OOHads"]]
deltaZPE[3] = 2.0*zpe[species["H2O"]] - 1.5*zpe[species["H2"]] - zpe[species["OOHads"]]

# shift potential shift, added to the right-hand side
shift = np.zeros(rxn_num)
shift[3] = 4.92  # OER4

# reation energies (in eV) and equilibrium constant
df_reac  = pd.read_json(reac_json)
df_reac  = df_reac.set_index("unique_id")
num_data = len(df_reac)

for id in range(num_data):
    if "score" in df_reac.iloc[id].index:
    # already calculated
    	pass

    if isinstance(df_reac.iloc[id].reaction_energy, list):
        unique_id = df_reac.iloc[id].name
        deltaE = df_reac.iloc[id].reaction_energy
        deltaE = np.array(deltaE)
        deltaH = deltaE + deltaZPE
        deltaG = deltaH - T*deltaS
        deltaG = deltaG + shift

        eta_oer  = np.max(deltaG[0:4]) - E_redox["OER"]  # overpotential of OER

        score = -eta_oer

        data = {"unique_id": unique_id, "reaction_energy": list(deltaE),
                "gibbs_energy": list(deltaG), "temperature": T, "score": score}

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
