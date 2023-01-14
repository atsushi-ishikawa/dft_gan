import sys
import numpy as np
import pandas as pd
import argparse
import json
from reaction_tools import get_number_of_reaction
import math

np.set_printoptions(precision=3)

parser = argparse.ArgumentParser()
parser.add_argument("--reac_json", default="reaction_energy.json", help="json for reading rxn. energy and writing rate")

args = parser.parse_args()
reac_json = args.reac_json

debug = False

T = 700  # K
ptot = 100.0  # bar
kJtoeV = 1/98.415
kB = 8.617e-5  # eV/K
RT = 8.314*1.0e-3*T*kJtoeV  # J/K -> kJ/K * K -> eV

gas = {"N2": 0, "H2":1, "NH3": 2}
ads = {"N" : 0, "H":1, "NH": 2, "NH2": 3, "NH3": 4, "vac": 5}

# --------------------------
reactionfile = "nh3.txt"
rxn_num = get_number_of_reaction(reactionfile)

p = np.zeros(len(gas))
conversion = 0.1
p[gas["N2"]]  = ptot*(1.0/4.0)
p[gas["H2"]]  = ptot*(3.0/4.0)
p[gas["NH3"]] = p[gas["N2"]]*conversion

# coverage
theta = np.zeros(len(ads))

# entropy in eV/K
# loss in each elementary reaction
deltaS    = np.zeros(rxn_num)
deltaS[0] = -1.98e-3  # N2
deltaS[1] = -1.35e-3  # H2
deltaS[5] =  2.00e-3  # NH3 

# zero-point energy (ZPE) in eV
# loss in each elementary reaction
deltaZPE    = np.zeros(rxn_num)
deltaZPE[0] = -0.145  # N2
deltaZPE[1] = -0.258  # H2
deltaZPE[5] =  0.895  # NH3 

# thermal correction (translation + rotation) in eV
# loss in each elementary reaction
deltaTherm    = np.zeros(rxn_num)
deltaTherm[0] = -(3/2)*RT - RT # N2
deltaTherm[1] = -(3/2)*RT - RT # H2
deltaTherm[5] =  (3/2)*RT + RT # NH3 

# pressure correction i.e. deltaG = deltaG^0 + RT*ln(p/p0)
# loss in each elementary reaction
RTlnP = np.zeros(rxn_num)
RTlnP[0] -= RT*np.log(p[gas["N2"]])
RTlnP[1] -= RT*np.log(p[gas["H2"]])
RTlnP[5] += RT*np.log(p[gas["NH3"]])

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

        deltaH  = deltaE + deltaZPE + deltaTherm

        deltaG  = deltaH - T*deltaS
        deltaG += RTlnP

        K = np.exp(-deltaG/(kB*T))

        # rate-determining step
        rds = 0
        # activation energy
        # Bronsted-Evans-Polanyi --- universal a and b for stepped surface (Norskov 2002)
        alpha = 0.87
        beta  = 1.34
        tmp = alpha*deltaE + beta
        Ea  = tmp[rds]

        A  = 1.2e6 / np.sqrt(T)  # Dahl J.Catal., converted from bar^-1 to Pa^-1
        #A  = 0.241 # [s^-1] Logadottir, J.Catal 220 273 2003
        k  = A*np.exp(-Ea/RT)

        # coverage
        tmp = 1 + np.sqrt(K[1]*p[gas["H2"]]) \
                + p[gas["NH3"]]/(np.sqrt(K[1]*p[gas["H2"]])*K[4]*K[5]) \
                + p[gas["NH3"]]/(K[1]*p[gas["H2"]]*K[3]*K[4]*K[5]) \
                + p[gas["NH3"]]/(K[1]**(3/2)*p[gas["H2"]]**(3/2)*K[2]*K[3]*K[4]*K[5]) \
                + p[gas["NH3"]]/K[5]

        theta[ads["vac"]] = 1/tmp
        theta[ads["H"]]   = np.sqrt(K[1]*p[gas["H2"]])*theta[ads["vac"]]
        theta[ads["NH3"]] = (p[gas["NH3"]]/K[5])*theta[ads["vac"]]
        theta[ads["NH2"]] = (p[gas["NH3"]]/(np.sqrt(K[1]*p[gas["H2"]])*K[4]*K[5]))*theta[ads["vac"]]
        theta[ads["NH"]]  = (p[gas["NH3"]]/(K[1]*p[gas["H2"]]*K[3]*K[4]*K[5]))*theta[ads["vac"]]
        theta[ads["N"]]   = (p[gas["NH3"]]/(K[1]**(3/2)*p[gas["H2"]]**(3/2)*K[2]*K[3]*K[4]*K[5]))*theta[ads["vac"]]

        Keq     = K[0]*(K[1]**3)*K[2]**2*K[3]**2*K[4]**2*K[5]**2
        gamma   = (1/Keq)*(p[gas["NH3"]]**2/(p[gas["N2"]]*p[gas["H2"]]**3))
        rate    = k*p[gas["N2"]]*theta[ads["vac"]]**2*(1-gamma) # maybe TOF

        score   = np.log10(rate)

        data = {"unique_id": unique_id, "reaction_energy": list(deltaE), "coverage": list(theta),
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
            with open(reac_json, "w") as f:
                json.dump(datum, f, indent=4)

        if debug:
            print("K[0]={0:4.2e}, K[1]={1:4.2e}, K[2]={2:4.2e}, K[3]={3:4.2e}, K[4]={4:4.2e}, K[5]={5:4.2e}".format(K[0],K[1],K[2],K[3],K[4],K[5]))
            print("theta[N]={0:4.2e}, theta[H]={1:4.2e}, theta[NH]={2:4.2e}, theta[NH2]={3:4.2e}, theta[NH3]={4:4.2e}, theta[vac]={5:4.2e}"
                    .format(theta[ads["N"]], theta[ads["H"]], theta[ads["NH"]], theta[ads["NH2"]], theta[ads["NH3"]], theta[ads["vac"]]))
            print("Ea = {0:5.3f} eV, score = {1:5.3e}".format(Ea, score))
    else:
        print("reaction energy not available")

