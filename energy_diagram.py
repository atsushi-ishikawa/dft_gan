import sys
import numpy as np
import pandas as pd
import argparse
import json
from scipy import interpolate
import matplotlib.pyplot as plt
import seaborn as sns
from tools import find_highest

parser = argparse.ArgumentParser()
parser.add_argument("--id", default="")
parser.add_argument("--reac_json", default="reaction_energy.json", help="json for reading rxn. energy and writing rate")
parser.add_argument("--best", default=True)

args = parser.parse_args()
reac_json = args.reac_json
best = args.best

if best:
	unique_id = find_highest(json=reac_json, score="rate")
else:
	unique_id = args.id

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

rds = 1
y1  = deltaE
y1  = np.insert(y1, 0, 0.0)
num_rxn = len(y1)+1
Ea  = y1[rds] + 0.2
y1  = np.insert(y1, rds, y1[rds-1])

x1 = np.arange(0, num_rxn)
x1_latent = np.linspace(0, num_rxn-1, 100)
f1 = interpolate.interp1d(x1, y1, kind="nearest")

x2 = [rds-0.5, rds, rds+0.5]
x2 = np.array(x2)
y2 = np.array([y1[rds-1], Ea, y1[rds+1]])
f2 = interpolate.interp1d(x2, y2, kind="quadratic")

y = np.array([])
for i in x1_latent:
	val1 = f1(i)
	val2 = -1.0e-10
	try:
		val2 = f2(i)
	except:
		pass
	y = np.append(y, max(val1, val2))

sns.set(style="darkgrid", rc={"lines.linewidth": 2.0, "figure.figsize": (10, 4)})
p = sns.lineplot(x=x1_latent, y=y, sizes=(0.5, 1.0))
p.set_xlabel("Steps")
p.set_ylabel("Energy (eV)")

filename = unique_id + "_" + "ped.png"
plt.savefig(filename)
