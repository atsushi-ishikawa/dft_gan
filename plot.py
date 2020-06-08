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

maxrun = 3
vals = []
for i in range(maxrun):
    run = df[df["run"]==i]

    val = np.sort(run["reaction_energy"].dropna().values)
    vals.append(val)

for i in range(maxrun):
    val = vals[i]
    label = "%d" % i
    plt.bar(range(len(val)), val, label=label)
    plt.legend()

plt.show()
