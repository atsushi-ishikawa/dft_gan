import plotly.graph_objects as go
import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
from tools import load_ase_json
import h5py

def get_colorpalette(colorpalette, n_colors):
	palette = sns.color_palette(colorpalette, n_colors)
	rgb = ['rgb({},{},{})'.format(*[x*256 for x in rgb]) for rgb in palette]
	return rgb

cwd       = os.path.abspath(__file__)
dirname   = os.path.dirname(cwd)
surf_json = os.path.join(dirname + "/surf.json")
reac_json = os.path.join(dirname + "/reaction_energy.json")
loss_file = os.path.join(dirname + "/loss.h5")
eneg_file = os.path.join(dirname + "/ped.h5")

interval = 60*(60*1e3)  # in milisec
height = 320  # height of each figure
#
# plotting currently best energy diagram
#
h5file = h5py.File(eneg_file, "r")
x = h5file["x"][:]
y = h5file["y"][:]

figure = go.Figure()
figure.add_trace(go.Scatter(x=x, y=y, mode="lines"))
figure.update_layout(
					xaxis = dict(showline=True, showgrid=False, showticklabels=True, linecolor="black", linewidth=2, tickformat="",
								ticks="outside", tickfont=dict(family="Arial", size=16, color="black"), mirror=True),
					yaxis = dict(showline=True, showgrid=False, showticklabels=True, linecolor="black", linewidth=2, tickformat=".1f",
								ticks="outside", tickfont=dict(family="Arial", size=16, color="black"), mirror=True),
					plot_bgcolor="white",
					)
figure.update_xaxes(title="Steps", title_font_family="Arial", color="black", title_font_size=18, tickwidth=2, ticklen=8)
figure.update_yaxes(title="Reaction energy (eV)", title_font_family="Arial", color="black", title_font_size=18, tickwidth=2, ticklen=8, range=[-3,1])
figure.write_image("ped.pdf")
#
# score bar plot from json file
#
df1 = load_ase_json(surf_json)
df2 = pd.read_json(reac_json)

df1 = df1.set_index("unique_id")
df2 = df2.set_index("unique_id")

df = pd.concat([df1, df2], axis=1)
df = df[df["score"].notna()]

runmax = df["run"].max()
runmin = df["run"].min()
maxval = df["score"].max()
minval = df["score"].min()

if abs(maxval) < 1.0:
	maxval = 2.0
elif maxval < 0.0:
	maxval = 0.0

df = df.sort_values("score", ascending=False)
df = df.reset_index()

colors = get_colorpalette("viridis", runmax+1)

figure = go.Figure()
for i in range(runmax + 1):
	df_now = df[df["run"]==i]
	x = df_now.index.values
	y = df_now["score"]
	color = "crimson" if i==runmax else colors[i]
	opacity = 0.2 if i==0 else 1.0
	figure.add_trace(go.Bar(x=x, y=y, marker_color=color, opacity=opacity, 
						   customdata=df_now[["score", "chemical_formula", "unique_id"]], 
						   name="run " + str(i)))

figure.update_layout(plot_bgcolor="white")
figure.update_xaxes(title="", title_font_family="Arial", color="black", title_font_size=18, tickwidth=2, ticklen=8,
					showline=True, showgrid=False, showticklabels=True, linecolor="black", linewidth=2, tickformat="",
					ticks="outside", tickfont=dict(family="Arial", size=16, color="black"), mirror=True)
figure.update_yaxes(title="Score", title_font_family="Arial", color="black", title_font_size=18, tickwidth=2, ticklen=8,
					showline=True, showgrid=False, showticklabels=True, linecolor="black", linewidth=2, tickformat=".1f",
					ticks="outside", tickfont=dict(family="Arial", size=16, color="black"), mirror=True, range=[minval-0.05*abs(minval), maxval])
figure.write_image("bar.pdf")
#
# loss plot
#
h5file = h5py.File(loss_file, "r")
epoch  = h5file["epoch"][:]
D_loss = h5file["D_loss"][:]
G_loss = h5file["G_loss"][:]

figure = go.Figure()
figure.add_trace(go.Scatter(x=epoch, y=D_loss, mode="lines", name="Discriminator loss"))
figure.add_trace(go.Scatter(x=epoch, y=G_loss, mode="lines", name="Generator loss"))
figure.update_xaxes(title="Epoch", title_font_family="Arial", color="black", title_font_size=18, tickwidth=2, ticklen=8,
					showline=True, showgrid=False, showticklabels=True, linecolor="black", linewidth=2, tickformat="",
					ticks="outside", tickfont=dict(family="Arial", size=16, color="black"), mirror=True)
figure.update_yaxes(title="Loss", title_font_family="Arial", color="black", title_font_size=18, tickwidth=2, ticklen=8,
					showline=True, showgrid=False, showticklabels=True, linecolor="black", linewidth=2, tickformat=".1f",
					ticks="outside", tickfont=dict(family="Arial", size=16, color="black"), mirror=True, range=[0,5])
figure.write_image("loss.pdf")
#
# coverage
#
df_reac = pd.read_json(reac_json)
df_reac = df_reac.set_index("unique_id")
df_reac = df_reac.sort_values("score", ascending=False)
species  = df_reac.iloc[0]["species"]
coverage = df_reac.iloc[0]["coverage"]

figure = go.Figure()
figure.add_trace(go.Bar(x=species, y=coverage, marker_color="cadetblue", marker_line_width=1, marker_line_color="black"))
figure.update_layout(plot_bgcolor="white")
figure.update_xaxes(title="Species", title_font_family="Arial", color="black", title_font_size=18, tickwidth=2, ticklen=8,
					showline=True, showgrid=False, showticklabels=True, linecolor="black", linewidth=2, tickformat="",
					ticks="outside", tickfont=dict(family="Arial", size=16, color="black"), mirror=True)
figure.update_yaxes(title="Fractional coverage (-)", title_font_family="Arial", color="black", title_font_size=18, tickwidth=2, ticklen=8,
					showline=True, showgrid=False, showticklabels=True, linecolor="black", linewidth=2, exponentformat="power",
					ticks="outside", tickfont=dict(family="Arial", size=16, color="black"), mirror=True, type="log")
figure.write_image("coverage.pdf")
#
# reaction energy
#
deltaE = df_reac.iloc[0]["reaction_energy"]
x = list(range(len(deltaE)))
x = ["N form", "H form", "NH form", "NH2 form", "NH3 form", "NH3 desorp"] 

figure = go.Figure()
figure.add_trace(go.Bar(x=x, y=deltaE, marker_color="cadetblue", marker_line_width=1, marker_line_color="black"))
figure.update_layout(plot_bgcolor="white")
figure.update_xaxes(title="Species", title_font_family="Arial", color="black", title_font_size=18, tickwidth=2, ticklen=8,
					showline=True, showgrid=False, showticklabels=True, linecolor="black", linewidth=2, tickformat="",
					ticks="outside", tickfont=dict(family="Arial", size=16, color="black"), mirror=True)
figure.update_yaxes(title="Reaction energy (eV)", title_font_family="Arial", color="black", title_font_size=18, tickwidth=2, ticklen=8,
					showline=True, showgrid=False, showticklabels=True, linecolor="black", linewidth=2, tickformat=".1f",
					ticks="outside", tickfont=dict(family="Arial", size=16, color="black"), mirror=True, zeroline=True, zerolinecolor="black")
figure.write_image("reaction_energy.pdf")

#
# violin plot for each iterations
#
figure = go.Figure()
for run in range(runmax+1):
 	figure.add_trace(go.Violin(x=df["run"][df["run"]==run], y=df["score"][df["run"]==run], 
 					box=dict(visible=True), points="all", fillcolor="cadetblue", line_color="black", opacity=0.6, name="run "+str(run)))

figure.update_layout(plot_bgcolor="white")
figure.update_xaxes(title="Run", title_font_family="Arial", color="black", title_font_size=18, tickwidth=2, ticklen=8,
					showline=True, showgrid=False, showticklabels=True, linecolor="black", linewidth=2, tickformat="",
					ticks="outside", tickfont=dict(family="Arial", size=16, color="black"), mirror=True)
figure.update_yaxes(title="Score", title_font_family="Arial", color="black", title_font_size=18, tickwidth=2, ticklen=8,
					showline=True, showgrid=False, showticklabels=True, linecolor="black", linewidth=2, tickformat=".1f",
					ticks="outside", tickfont=dict(family="Arial", size=16, color="black"), mirror=True)
figure.write_image("violin.pdf")
#
figure = go.Figure()
means = []
for run in range(runmax+1):
	means.append(df["score"][df["run"]<=run].mean())

figure = go.Figure()
figure.add_trace(go.Scatter(x=list(range(runmax+1)), y=means, marker=dict(size=10)))

figure.update_layout(plot_bgcolor="white")
figure.update_xaxes(title="Run", title_font_family="Arial", color="black", title_font_size=18, tickwidth=2, ticklen=8,
					showline=True, showgrid=False, showticklabels=True, linecolor="black", linewidth=2, tickformat="",
					ticks="outside", tickfont=dict(family="Arial", size=16, color="black"), mirror=True)
figure.update_yaxes(title="Mean score", title_font_family="Arial", color="black", title_font_size=18, tickwidth=2, ticklen=8,
					showline=True, showgrid=False, showticklabels=True, linecolor="black", linewidth=2, tickformat=".1f",
					ticks="outside", tickfont=dict(family="Arial", size=16, color="black"), mirror=True)
figure.write_image("mean.pdf")
#
# reaction energy and coverage for selected case
#
runs = [0, 1, 2]

x = ["N form", "H form", "NH form", "NH2 form", "NH3 form", "NH3 desorp"] 
figure = go.Figure()
for run in runs:
	idx = df[df["run"]==run]["score"].idxmax()
	deltaE = df.loc[idx]["reaction_energy"]
	figure.add_trace(go.Bar(x=x, y=deltaE, marker_line_width=1, marker_line_color="black", name="run "+str(run)))

figure.update_layout(plot_bgcolor="white")
figure.update_xaxes(title="Species", title_font_family="Arial", color="black", title_font_size=18, tickwidth=2, ticklen=8,
					showline=True, showgrid=False, showticklabels=True, linecolor="black", linewidth=2, tickformat="",
					ticks="outside", tickfont=dict(family="Arial", size=16, color="black"), mirror=True)
figure.update_yaxes(title="Reaction energy (eV)", title_font_family="Arial", color="black", title_font_size=18, tickwidth=2, ticklen=8,
					showline=True, showgrid=False, showticklabels=True, linecolor="black", linewidth=2, tickformat=".1f",
					ticks="outside", tickfont=dict(family="Arial", size=16, color="black"), mirror=True, zeroline=True, zerolinecolor="black")
figure.write_image("reaction_energy2.pdf")

# coverage
species  = df_reac.iloc[0]["species"]
figure = go.Figure()
for run in runs:
	idx = df[df["run"]==run]["score"].idxmax()
	coverage = df.loc[idx]["coverage"]
	figure.add_trace(go.Bar(x=species, y=coverage, marker_line_width=1, marker_line_color="black", name="run "+str(run)))

figure.update_layout(plot_bgcolor="white")
figure.update_xaxes(title="Species", title_font_family="Arial", color="black", title_font_size=18, tickwidth=2, ticklen=8,
					showline=True, showgrid=False, showticklabels=True, linecolor="black", linewidth=2, tickformat="",
					ticks="outside", tickfont=dict(family="Arial", size=16, color="black"), mirror=True)
figure.update_yaxes(title="Fractional coverage (-)", title_font_family="Arial", color="black", title_font_size=18, tickwidth=2, ticklen=8,
					showline=True, showgrid=False, showticklabels=True, linecolor="black", linewidth=2, exponentformat="power",
					ticks="outside", tickfont=dict(family="Arial", size=16, color="black"), mirror=True, type="log")
figure.write_image("coverage2.pdf")

