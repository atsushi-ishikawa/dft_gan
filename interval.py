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
eneg_file = os.path.join(dirname + "/diag.h5")

interval = 60*(60*1e3)  # in milisec
height = 320  # height of each figure

external_stylesheets = [
	'https://codepen.io/chriddyp/pen/bWLwgP.css',
	{
		'href': 'https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css',
		'rel': 'stylesheet',
		'integrity': 'sha384-MCw98/SFnGE8fJT3GXwEOngsV7Zt27NXFoaoApmYm81iuXoPkFOJwJ8ERdknLPMO',
		'crossorigin': 'anonymous'
	}
]
app = dash.Dash(__name__, requests_pathname_prefix="/nn_reac/", external_stylesheets=external_stylesheets)

app.layout = html.Div(
	children=[
		html.Div(children=[
			html.Div(id="status", className="header-description"),
		], className="header"),
		html.Div(children=[
			html.Div(id="the number"),
			dcc.Interval(id="interval-component", interval=interval, n_intervals=0),
		]),
		dcc.Graph(id="graph_bar"),
		html.Div([
			html.Div([
				dcc.Graph(id="graph_loss"),
			], className="six columns"),
			html.Div([
				dcc.Graph(id="energy"),
			], className="six columns")
		], className="row"),
#		html.Table([
#			html.Tr([
#				html.Td([dcc.Graph(id="graph_loss")]),
#				#html.Td([dcc.Graph(id="energy")]),
#				#html.Td([html.Div(id="structure")]),
#			]),
#		]),
	],
)

#
# counter
#
@app.callback(Output("the number", "children"),
	[Input("interval-component", "n_intervals"),
	 Input("interval-component", "interval")])
def display_num(n_intervals, intervals):
	style = {"padding": "5px", "fontsize": "40px"}
	return html.Div('updated {0:d} times (updating every {1:.0f} min)'.format(n_intervals, intervals/60/1e3), style=style)

#
# getting current process
#
@app.callback(Output("status", "children"),
	[Input("interval-component", "n_intervals")])
def getting_status(n):
	if os.path.exists(dirname + "/doing_GAN"):
		status = "Now doing GAN"
	elif os.path.exists(dirname + "/doing_reaction_energy_calc"):
		status = "Now doing reaction energy calculation"
	elif os.path.exists(dirname + "/doing_finished"):
		status = "Finished"
	else:
		status = "doing something else"

	return html.H2("%s" % status)

#
# plotting currently best energy diagram
#
@app.callback(Output("energy", "figure"),
			 [Input("interval-component", "n_intervals")])
def make_energy_diagram(n):
	h5file = h5py.File(eneg_file, "r")
	x = h5file["x"][:]
	y = h5file["y"][:]

	figure = go.Figure()
	figure.add_trace(go.Scatter(x=x, y=y, mode="lines"))
	figure.update_layout(margin=dict(r=40, t=20, b=20),
						 xaxis_title="steps", yaxis_title="Potential energy (eV)",
						 height=height)
	return figure
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
# score bar plot from json file
#
@app.callback(Output("graph_bar", "figure"),
             [Input("interval-component", "n_intervals")])
def make_score_bar(n):
	score = "rate"
	df1 = load_ase_json(surf_json)
	df2 = pd.read_json(reac_json)

	df1 = df1.set_index("unique_id")
	df2 = df2.set_index("unique_id")

	df = pd.concat([df1, df2], axis=1)
	df = df[df[score].notna()]

	runmax = df["run"].max()
	runmin = df["run"].min()
	maxval = df[score].max()
	minval = df[score].min()

	df = df.sort_values(score, ascending=False)
	df = df.reset_index()

	colors = get_colorpalette("viridis", runmax+1)

	figure = go.Figure()
	for i in range(runmax + 1):
		df_now = df[df["run"]==i]
		x = df_now.index.values
		y = df_now[score]
		color = "crimson" if i==runmax else colors[i]
		opacity = 0.2 if i==0 else 1.0

		figure.add_trace(go.Bar(x=x, y=y, marker_color=color, opacity=opacity, 
							   customdata=df_now[["chemical_formula","unique_id"]], 
							   hovertemplate="%{customdata[0]}<br>%{customdata[1]}",
							   name="run " + str(i)))

	#figure.update_yaxes(range=[minval-0.01, maxval+0.01])
	figure.update_layout(margin=dict(r=20, t=20, b=20),
						 legend=dict(orientation="h", yanchor="bottom", y=1.02),
						 yaxis_title=score,
						 height=height)
	return figure

#
# loss plot
#
@app.callback(Output("graph_loss", "figure"),
             [Input("interval-component", "n_intervals")])
def make_loss_figure(n):
	h5file = h5py.File(loss_file, "r")
	epoch  = h5file["epoch"][:]
	D_loss = h5file["D_loss"][:]
	G_loss = h5file["G_loss"][:]

	figure = go.Figure()
	figure.add_trace(go.Scatter(x=epoch, y=D_loss, mode="lines", name="Discriminator loss"))
	figure.add_trace(go.Scatter(x=epoch, y=G_loss, mode="lines", name="Generator loss"))
	figure.update_layout(margin=dict(r=20, t=20, b=20),
						 xaxis_title="epoch", yaxis_title="loss",
						 legend=dict(orientation="h", yanchor="bottom", y=1.02),
						 height=height)
	return figure


if __name__ == "__main__":
	app.run_server()

