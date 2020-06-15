import pandas as pd
import matplotlib.pyplot as plt
from tools import load_ase_json
import numpy as np

surf_json = "surf.json"
reac_json = "reaction_energy.json"

df1 = load_ase_json(surf_json)
df2 = pd.read_json(reac_json)

df1 = df1.set_index("unique_id")
df2 = df2.set_index("unique_id")

df = pd.concat([df1, df2], axis=1)
df = df[df["reaction_energy"].notna()]

runmax = df["run"].max()
runmin = df["run"].min()
maxval = df["reaction_energy"].max()
minval = df["reaction_energy"].min()

df3 = df.sort_values("reaction_energy")
df3 = df3.reset_index()


def norm(run):
    return (run - runmin) / (runmax - runmin)


plt.figure(figsize=(18, 6))
colormap = plt.cm.viridis_r
for i in range(runmax+1):
    current = df3[df3["run"] == i]
    pos = current.index.values
    val = current.reaction_energy
    color = "grey" if i == 0 else colormap(norm(i))
    label = "run %2d" % i
    plt.bar(pos, val, color=color, label=label, width=0.8, alpha=0.5)

plt.ylim([minval-0.01, maxval+0.01])
plt.legend()
plt.show()
