import os
import pandas as pd
from tools import load_ase_json

todo_list = "todolist.txt"
surf_json = "surf.json"
reac_json = "reaction_energy.json"

print("making to-do list ... result will be stored in {}".format(todo_list))

df1 = load_ase_json(surf_json)

if os.path.exists(reac_json):
    df2 = pd.read_json(reac_json)
else:
    df2 = pd.DataFrame(columns=["unique_id"])

df1 = df1.set_index("unique_id")
df2 = df2.set_index("unique_id")
df = pd.concat([df1, df2], axis=1)

score = "score"
if score in df.columns:
    df = df[df[score].isnull()]
else:
    # first time
    pass

not_done = df.index.values

if len(not_done) == 0:
    # all done
    pass
else:
    # make todo_list
    with open(todo_list, "w") as f:
        for i in not_done:
            f.write(i+"\n")
