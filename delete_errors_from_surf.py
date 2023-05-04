import os
import numpy as np
import pandas as pd
import argparse
import json
import shutil
from tools import load_ase_json
from ase.db import connect

parser = argparse.ArgumentParser()
parser.add_argument("--surf_json", default="surf.json", help="json for surfaces")
parser.add_argument("--reac_json", default="reaction_energy.json", help="json for reaction energies")
parser.add_argument("--target", default="score", help="target key name")

args = parser.parse_args()
surf_json = args.surf_json
reac_json = args.reac_json
target = args.target

# take copy
shutil.copyfile(surf_json, surf_json.split(".")[0] + "_raw.json")
shutil.copyfile(reac_json, reac_json.split(".")[0] + "_raw.json")

print("deleting unfinished result from {}".format(surf_json))

df_surf = load_ase_json(surf_json)

if os.path.exists(reac_json):
    df_reac = pd.read_json(reac_json)
else:
    df_reac = pd.DataFrame(columns=["unique_id"])

df_surf = df_surf.set_index("unique_id")
df_reac = df_reac.set_index("unique_id")
df = pd.concat([df_surf, df_reac], axis=1)

# delete null
null_list = df[df[target].isnull()].index.values

# delete errournous value
thre = -1.5  # -1.5 or -2.5, -2.5 found several outliers.
bad_list = df[df[target] < thre].index.values
del_list = np.concatenate([null_list, bad_list])

print("deleting outliers ({0:d} samples with score < {1:5.3f})".format(len(del_list), thre))

#
## delete from surface json file
db_surf = connect(surf_json)
for i in del_list:
    id = db_surf.get(unique_id=i).id
    db_surf.delete([id])

# delete from reacion energy json file
df_reac = df_reac[df_reac["score"] > thre]
df_reac["unique_id"] = df_reac.index
df_reac = df_reac[df_reac.columns[::-1]]  # move unique_id to first
parsed = json.loads(df_reac.to_json(orient="records"))
with open(reac_json, "w") as f:
    json.dump(parsed, f, indent=4)

