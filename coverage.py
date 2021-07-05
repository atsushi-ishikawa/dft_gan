import os,sys
import numpy as np
import pandas as pd
import argparse
import json
from tools import find_highest

reac_json = "reaction_energy.json"

df_reac  = pd.read_json(reac_json)
df_reac  = df_reac.set_index("unique_id")
df_reac  = df_reac.sort_values("score", ascending=False)
coverage = df_reac.iloc[0]["coverage"]
print(coverage)

#if unique_id in df_reac.index:
#	try:
#		deltaE = df_reac.loc[unique_id].reaction_energy
#	except:
#		print("reaction energy not found on %s" % reac_json)
#		sys.exit(1)

import matplotlib.pyplot as plt
import seaborn as sns

df_now = pd.DataFrame({"steps":list(range(len(coverage))), "coverage": coverage})
sns.set(style="darkgrid")
p = sns.barplot(x="steps", y="coverage", data=df_now, color="steelblue")
plt.savefig("asdf.png")

