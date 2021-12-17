import json
import pandas as pd
#
# merge two json files and convert it to csv
#
tmpdf = json.load(open("surf.json", "r"))
df_surf = pd.DataFrame()
for i in range(len(tmpdf)-2):
	tmp = pd.json_normalize(tmpdf[str(i+1)])
	df_surf = pd.concat([df_surf, tmp])

df_reac = json.load(open("reaction_energy.json", "r"))
df_reac = pd.json_normalize(df_reac)

df_surf = df_surf.set_index("unique_id")
df_reac = df_reac.set_index("unique_id")

df = pd.concat([df_surf, df_reac], axis=1)

df.to_csv("gan_data.csv")

