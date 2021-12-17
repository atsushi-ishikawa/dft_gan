import numpy as np
import seaborn
import json
import pandas as pd
import matplotlib.pyplot as plt
import argparse

plt.rcParams["figure.figsize"] = (14, 10)

parser = argparse.ArgumentParser()
parser.add_argument("--jsonfile", default="surf.json", help="json file for surface")
args = parser.parse_args()
jsonfile = args.jsonfile

with open(jsonfile, "r") as f:
	df = json.load(f)

atomic_numbers = []
last = df["nextid"]
for i in range(1, last):
	atomic_number = df[str(i)]["data"]["atomic_numbers"]
	atomic_numbers.append(atomic_number)

fig, ax = plt.subplots()
cmap = "YlGn_r"
seaborn.heatmap(atomic_numbers, linewidth=0.003, linecolor="white", cmap=cmap, xticklabels=1, yticklabels=1)
plt.tight_layout()
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.show()

