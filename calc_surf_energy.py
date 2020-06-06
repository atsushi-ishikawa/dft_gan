from ase import Atoms, Atom
from ase.build import fcc111
from ase.visualize import view
from ase.calculators.emt import EMT
from ase.db import connect
from ase.optimize import BFGS
import numpy as np
import os, sys, json
import random

injson  = "surf.json"
outjson = "surface_energy.json"

db1 = connect(injson)
db2 = connect(outjson)
calc = EMT()

numdata = db1.count()+1

datum = []
for id in range(1,numdata):
	surf = db1.get_atoms(id=id)
	obj  = db1[id]
	data = obj.data
	unique_id = obj["unique_id"]

	surf.set_calculator(calc)

	# optimize
	opt = BFGS(surf)
	opt.run(fmax=0.1)

	# calc energy
	Etot = surf.get_potential_energy()

	data = {"unique_id" : unique_id, "total_energy":Etot}
	datum.append(data)

with open(outjson,"w") as f:
	json.dump(datum, f, indent=4)
