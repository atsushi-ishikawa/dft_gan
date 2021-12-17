import plotly.graph_objects as go
import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
from tools import load_ase_json
import h5py
import argparse

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

# argument
parser = argparse.ArgumentParser()
parser.add_argument("--runmax", default=0,   type=int, help="maximum number of run")
parser.add_argument("--runs",   default=[0], type=int, nargs="+", help="which runs do you want to plot?")
parser.add_argument("--height", default=500, type=int, help="height of figures")

args   = parser.parse_args()
runmax = args.runmax
runs   = args.runs
height = args.height

#
# load data into DataFrame
#   1) surf.json --> df_surf
#   2) reaction_energy.json --> df_reac
#
df1 = load_ase_json(surf_json)
df2 = pd.read_json(reac_json)

df1 = df1.set_index("unique_id")
df2 = df2.set_index("unique_id")

df = pd.concat([df1, df2], axis=1)
df = df[df["score"].notna()]

if runmax == 0:
	runmax = df["run"].max()
runmin = df["run"].min()
maxval = df["score"].max()
minval = df["score"].min()

if abs(maxval) < 1.0:
	maxval = 2.0
elif maxval < 0.0:
	maxval = 0.0

df = df.sort_values("score", ascending=False)
df_surf = df.reset_index()

df_reac = pd.read_json(reac_json)
df_reac = df_reac.set_index("unique_id")
df_reac = df_reac.sort_values("score", ascending=False)

# get color palette
colors = get_colorpalette("viridis", runmax+2)
#colors = get_colorpalette("terrain", runmax+2)
#
# plotting currently best energy diagram
#
os.system('python energy_diagram.py')
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
figure = go.Figure()
for i in range(runmax + 1):
	df_now = df_surf[df_surf["run"]==i]
	x = df_now.index.values
	y = df_now["score"]
	color = "crimson" if i==runmax else colors[i]
	opacity = 0.2 if i==0 else 1.0
	figure.add_trace(go.Bar(x=x, y=y, marker_color=color, opacity=opacity, 
						   customdata=df_now[["score", "chemical_formula", "unique_id"]], name="run " + str(i)))

figure.update_layout(plot_bgcolor="white", height=height, width=height*2)
figure.update_xaxes(title="", title_font_family="Arial", color="black", title_font_size=18, tickwidth=2, ticklen=8,
					showline=True, showgrid=False, showticklabels=True, linecolor="black", linewidth=2, tickformat="",
					ticks="outside", tickfont=dict(family="Arial", size=16, color="black"), mirror=True)
figure.update_yaxes(title="Score", title_font_family="Arial", color="black", title_font_size=18, tickwidth=2, ticklen=8,
					showline=True, showgrid=False, showticklabels=True, linecolor="black", linewidth=2, tickformat="",
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
figure.add_trace(go.Scatter(x=epoch, y=D_loss, mode="lines", name="Discriminator loss", marker_color="cadetblue", line_width=0.1))
figure.add_trace(go.Scatter(x=epoch, y=G_loss, mode="lines", name="Generator loss",     marker_color="royalblue", line_width=0.1))

figure.update_layout(plot_bgcolor="white", height=height)
figure.update_xaxes(title="Epoch", title_font_family="Arial", color="black", title_font_size=18, tickwidth=2, ticklen=8,
					showline=True, showgrid=False, showticklabels=True, linecolor="black", linewidth=2, tickformat="d",
					ticks="outside", tickfont=dict(family="Arial", size=16, color="black"), mirror=True)
figure.update_yaxes(title="Loss", title_font_family="Arial", color="black", title_font_size=18, tickwidth=2, ticklen=8,
					showline=True, showgrid=False, showticklabels=True, linecolor="black", linewidth=2, tickformat=".1f",
					ticks="outside", tickfont=dict(family="Arial", size=16, color="black"), mirror=True, range=[0,5])
figure.write_image("loss.pdf")
#
# violin plot for each iterations
#
figure = go.Figure()
for run in range(runmax+1):
 	figure.add_trace(go.Violin(x=df_surf["run"][df_surf["run"]==run], y=df_surf["score"][df_surf["run"]==run], 
 					box=dict(visible=True), points="all", fillcolor=colors[run+1], line_color="black", line_width=1, opacity=0.8, name="Run "+str(run)))

figure.update_layout(plot_bgcolor="white", height=height)
figure.update_xaxes(title="Run", title_font_family="Arial", color="black", title_font_size=18, tickwidth=2, ticklen=8,
					showline=True, showgrid=False, showticklabels=True, linecolor="black", linewidth=2, tickformat="",
					ticks="outside", tickfont=dict(family="Arial", size=16, color="black"), mirror=True)
figure.update_yaxes(title="Score", title_font_family="Arial", color="black", title_font_size=18, tickwidth=2, ticklen=8,
					showline=True, showgrid=False, showticklabels=True, linecolor="black", linewidth=2, tickformat=".1f",
					ticks="outside", tickfont=dict(family="Arial", size=16, color="black"), mirror=True)
figure.write_image("violin.pdf")
#
# mean plot
#
figure = go.Figure()
num = []
means = []
for run in range(runmax+1):
	num.append(df_surf["unique_id"][df_surf["run"]<=run].count())
	means.append(df_surf["score"][df_surf["run"]<=run].mean())

figure = go.Figure()
figure.add_trace(go.Scatter(x=num, y=means, marker=dict(size=10)))

figure.update_layout(plot_bgcolor="white", height=height)
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
x = ["N form", "H form", "NH form", "NH2 form", "NH3 form", "NH3 desorp"] 
figure = go.Figure()
for run in runs:
	idx = df_surf[df_surf["run"]==run]["score"].idxmax()
	deltaE = df_surf.loc[idx]["reaction_energy"]
	figure.add_trace(go.Bar(x=x, y=deltaE, marker_line_width=1, marker_color=colors[run+1], marker_line_color="black", name="Run "+str(run)))

figure.update_layout(plot_bgcolor="white", height=height, width=2*height)
figure.update_xaxes(title="Reactions", title_font_family="Arial", color="black", title_font_size=18, tickwidth=2, ticklen=8,
					showline=True, showgrid=False, showticklabels=True, linecolor="black", linewidth=2, tickformat="",
					ticks="outside", tickfont=dict(family="Arial", size=16, color="black"), mirror=True)
figure.update_yaxes(title="Reaction energy (eV)", title_font_family="Arial", color="black", title_font_size=18, tickwidth=2, ticklen=8,
					showline=True, showgrid=False, showticklabels=True, linecolor="black", linewidth=2, tickformat=".1f",
					ticks="outside", tickfont=dict(family="Arial", size=16, color="black"), mirror=True, zeroline=True, zerolinecolor="black", zerolinewidth=1)
figure.write_image("reaction_energy.pdf")

# coverage
species  = df_reac.iloc[0]["species"]
figure = go.Figure()
for run in runs:
	idx = df_surf[df_surf["run"]==run]["score"].idxmax()
	coverage = df_surf.loc[idx]["coverage"]
	figure.add_trace(go.Bar(x=species, y=coverage, marker_color=colors[run+1], marker_line_width=1, marker_line_color="black", name="Run "+str(run)))

figure.update_layout(plot_bgcolor="white", height=height, width=2*height)
figure.update_xaxes(title="Species", title_font_family="Arial", color="black", title_font_size=18, tickwidth=2, ticklen=8,
					showline=True, showgrid=False, showticklabels=True, linecolor="black", linewidth=2, tickformat="",
					ticks="outside", tickfont=dict(family="Arial", size=16, color="black"), mirror=True)
figure.update_yaxes(title="Fractional coverage (-)", title_font_family="Arial", color="black", title_font_size=18, tickwidth=2, ticklen=8,
					showline=True, showgrid=False, showticklabels=True, linecolor="black", linewidth=2, exponentformat="power",
					ticks="outside", tickfont=dict(family="Arial", size=16, color="black"), mirror=True, type="log")
figure.write_image("coverage.pdf")
#
# plotting currently best energy diagram
#
figure = go.Figure()
for run in runs:
	idx = df_surf[df_surf["run"]==run]["score"].idxmax()
	unique_id = df_surf.loc[idx]["unique_id"]

	os.system('python energy_diagram.py --id=%s' % unique_id)
	h5file = h5py.File(eneg_file, "r")
	x = h5file["x"][:]
	y = h5file["y"][:]
	figure.add_trace(go.Scatter(x=x, y=y, mode="lines", marker_color=colors[run+1], name="Run "+str(run)))
	h5file.close()

figure.update_layout(plot_bgcolor="white", height=height, width=2*height)
figure.update_xaxes(title="Steps", title_font_family="Arial", color="black", title_font_size=18, tickwidth=2, ticklen=8,
					showline=True, showgrid=False, showticklabels=True, linecolor="black", linewidth=2, tickformat="",
					ticks="outside", tickfont=dict(family="Arial", size=16, color="black"), mirror=True)
figure.update_yaxes(title="Potential energy (eV)", title_font_family="Arial", color="black", title_font_size=18, tickwidth=2, ticklen=8,
					showline=True, showgrid=False, showticklabels=True, linecolor="black", linewidth=2, tickformat=".1f",
					ticks="outside", tickfont=dict(family="Arial", size=16, color="black"), mirror=True, range=[-4,2])
figure.write_image("ped2.pdf")

