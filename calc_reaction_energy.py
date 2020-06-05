from ase import Atoms, Atom
from ase.build import add_adsorbate
from ase.visualize import view
from ase.calculators.emt import EMT
from ase.db import connect
import numpy as np
import os, sys, json
import random

injson  = "surf.json"
outjson = "reaction_energy.json"

db1 = connect(injson)
db2 = connect(outjson)
calc = EMT()

numdata = db1.count()+1

datum = []
reac = Atoms("O2", [(0,0,0), (0,0,1.0)])
#
# reactant
#
reac.set_calculator(calc)
Ereac = reac.get_potential_energy()
#
# product
#
prod1 = Atoms("O", [(0,0,0)])
prod2 = Atoms("O", [(0,0,0)])
prod1.set_calculator(calc)
prod2.set_calculator(calc)
Eprod1 = prod1.get_potential_energy()
Eprod2 = prod2.get_potential_energy()

for id in range(1,numdata):
	surf = db1.get_atoms(id=id)
	obj  = db1[id]
	data = obj.data
	unique_id = obj["unique_id"]

	surf.set_calculator(calc)
	Esurf = surf.get_potential_energy()

	add_adsorbate(surf, prod1, offset=(0.3, 0.3), height=1.3)
	add_adsorbate(surf, prod2, offset=(0.6, 0.6), height=1.3)
	Eprod_surf = surf.get_potential_energy()
	Ereac = (Eprod_surf + Eprod1 + Eprod2) - (Esurf + Ereac)

	data = {"unique_id" : unique_id, "reaction_energy":Ereac}
	datum.append(data)

with open(outjson,"w") as f:
	json.dump(datum, f)
