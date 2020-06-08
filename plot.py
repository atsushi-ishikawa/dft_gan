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

values = []
color = ["grey", "blue", "green"]
for _, irow in df.iterrows():
    value = irow["reaction_energy"]
    run = irow["run"]
    values.append([value,color[run]])

values = sorted(values, key=lambda x: x[0])

value = list(map(lambda x: x[0], values))
color = list(map(lambda x: x[1], values))

plt.bar(range(len(value)), value, color=color, alpha=0.4)
plt.ylim([np.min(value)-0.05, np.min(value)+0.1])
plt.show()
