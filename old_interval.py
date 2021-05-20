import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
from tools import load_ase_json

def get_colorpalette(colorpalette, n_colors):
	palette = sns.color_palette(colorpalette, n_colors)
	rgb = ['rgb({},{},{})'.format(*[x*256 for x in rgb]) for rgb in palette]
	return rgb

cwd       = os.path.abspath(__file__)
dirname   = os.path.dirname(cwd)
surf_json = os.path.join(dirname + "/surf.json")
reac_json = os.path.join(dirname + "/reaction_energy.json")

app = dash.Dash(__name__, requests_pathname_prefix="/nn_reac/")

app.layout = html.Div(
	children=[
		html.Div(children=[
			html.Div(id="status", className="header-description"),
		], className="header"),
		html.Div(children=[
			html.Div(id="the number"),
			dcc.Interval(id="interval-component", interval=60*1e3, n_intervals=0),
		]),
		html.Div(id="bar"),
		html.Table([
			html.Tr([
				html.Td([html.Div(id="loss")]),
				html.Td([html.Div(id="structure")]),
			]),
		]),
	dcc.Graph(id="graph"),
	],
)

#
# counter
#
@app.callback(Output("the number", "children"),
	[Input("interval-component", "n_intervals"),
	 Input("interval-component", "interval")])
def display_num(n_intervals, intervals):
	style = {"padding": "5px", "fontsize": "16px"}
	return html.Div('"n_intervals"={} times, "updating every {} min'.format(n_intervals, intervals/60/1e3), style=style)

#
# reaction energy bar plot
#
@app.callback(Output("bar", "children"),
	[Input("interval-component", "n_intervals")])
def making_bar(n):
	bar = glob.glob(dirname + "/log/bars*.png")
	bar = sorted(bar)
	if len(bar)==0:
		barfile = ""
		pass
	else:
		barfile = bar[-1]
		os.system("rm %s 2> /dev/null" % (dirname + "/assets/bars*.png"))
		os.system("cp %s %s" % (barfile, dirname + "/assets"))

	return html.Img(src=app.get_asset_url(os.path.basename(barfile)), width="100%")

#
# generator and discriminator loss
#
@app.callback(Output("loss", "children"),
	[Input("interval-component", "n_intervals")])
def making_loss(n):
	loss = glob.glob(dirname + "/log/loss*.png")
	loss = sorted(loss)
	if len(loss)==0:
		lossfile = ""
		pass
	else:
		lossfile = loss[-1]
		os.system("rm %s 2> /dev/null" % (dirname + "/assets/loss*.png"))
		os.system("cp %s %s" % (lossfile, dirname + "/assets"))

	return html.Img(src=app.get_asset_url(os.path.basename(lossfile)), width="80%")

#
# getting current process
#
@app.callback(Output("status", "children"),
	[Input("interval-component", "n_intervals")])
def getting_status(n):
	if os.path.exists(dirname + "/doing_GAN"):
		status = "GAN"
	elif os.path.exists(dirname + "/doing_reaction_energy_calc"):
		status = "reaction_energy_calc"
	elif os.path.exists(dirname + "/doing_preparation"):
		status = "preparation"
	else:
		status = "something_else"

	return html.H2("Now doing %s ..." % status)

#
# plotting currently best structure
#
@app.callback(Output("structure", "children"),
	[Input("interval-component", "n_intervals")])
def plot_structure(n):
	figfile = os.path.join(dirname + "/log/structure.png")
	if os.path.exists(figfile):
		os.system("cp %s %s" % (figfile, dirname + "/assets"))
		return html.Img(src=app.get_asset_url(os.path.basename(figfile)), width="80%")
	else:
		return html.H2("not yet")

#
# plot from json file
#
@app.callback(Output("graph", "figure"),
             [Input("interval-component", "n_intervals")])
def making_figure(n):
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

	df = df.sort_values("reaction_energy")
	df = df.reset_index()

	colors = get_colorpalette("viridis", runmax+1)

	figure = go.Figure()
	for i in range(runmax + 1):
		now = df[df["run"]==i]
		x = now.index.values
		y = now.reaction_energy
		opacity = 0.2 if i==0 else 1.0

		figure.add_trace(go.Bar(x=x, y=y, marker_color=colors[i], opacity=opacity, name="run " + str(i)))

	figure.update_yaxes(range=[minval-0.01, maxval+0.01])
	return figure


if __name__ == "__main__":
	app.run_server()
