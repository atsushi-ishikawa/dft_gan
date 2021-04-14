import random
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import os
import glob

cwd = os.path.abspath(__file__)
dirname = os.path.dirname(cwd)

app = dash.Dash(__name__, requests_pathname_prefix="/nn_reac/")

app.layout = html.Div(
	children=[
		html.Div(children=[
				html.H1("GAN", className="header-title"),
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
			])
		])
	]
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
	os.system("rm %s 2> /dev/null" % (dirname + "/assets/bars*.png"))
	bar = glob.glob(dirname + "/log/bars*.png")
	bar = sorted(bar)
	if len(bar)==0:
		pass
	else:
		barfile = bar[-1]
		os.system("cp %s %s" % (barfile, dirname + "/assets"))

	return html.Img(src=app.get_asset_url(os.path.basename(barfile)), width="100%")

#
# generator and discriminator loss
#
@app.callback(Output("loss", "children"),
	[Input("interval-component", "n_intervals")])
def making_loss(n):
	os.system("rm %s 2> /dev/null" % (dirname + "/assets/loss*.png"))
	loss = glob.glob(dirname + "/log/loss*.png")
	loss = sorted(loss)
	if len(loss)==0:
		pass
	else:
		lossfile = loss[-1]
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


if __name__ == "__main__":
	app.run_server()
