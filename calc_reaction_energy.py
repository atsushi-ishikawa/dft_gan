from ase import Atoms, Atom
from ase.build import add_adsorbate
from ase.visualize import view
from ase.calculators.emt import EMT
from ase.db import connect
from ase.optimize import BFGS
import numpy as np
import os, sys, json

injson  = "surf.json"
outjson = "reaction_energy.json"

# remove old one
if os.path.exists(outjson):
	os.remove(outjson)

db1 = connect(injson)
db2 = connect(outjson)
calc = EMT()

numdata = db1.count()+1

datum = []

check = False
#
# reactant
#
reac = Atoms("O2", [(0,0,0), (0,0,1.1)])
reac.set_calculator(calc)
print(" --- calculating %s ---" % reac.get_chemical_formula())
opt = BFGS(reac)
opt.run(fmax=0.1)
Ereac = reac.get_potential_energy()
#
# product
#
prod1 = Atoms("O", [(0,0,0)])
prod2 = Atoms("O", [(0,0,0)])
prod1.set_calculator(calc)
prod2.set_calculator(calc)
print(" --- calculating %s ---" % prod1.get_chemical_formula())
print(" --- calculating %s ---" % prod2.get_chemical_formula())
opt = BFGS(prod1)
opt.run(fmax=0.1)
Eprod1 = prod1.get_potential_energy()
opt = BFGS(prod2)
opt.run(fmax=0.1)
Eprod2 = prod2.get_potential_energy()
Eproduct  = Eprod1 + Eprod2
#
# loop over surfaces
#
for id in range(1,numdata):
	surf = db1.get_atoms(id=id)
	obj  = db1[id]
	data = obj.data
	unique_id = obj["unique_id"]

	print(" --- calculating %s ---" % surf.get_chemical_formula())
	surf.set_calculator(calc)
	opt = BFGS(surf)
	opt.run(fmax=0.1)
	Esurf = surf.get_potential_energy()

	add_adsorbate(surf, prod1, offset=(0.3, 0.3), height=1.3)
	add_adsorbate(surf, prod2, offset=(0.6, 0.6), height=1.3)
	print(" --- calculating %s ---" % surf.get_chemical_formula())
	opt = BFGS(surf)
	opt.run(fmax=0.1)
	Eprod_surf = surf.get_potential_energy()

	Ereactant = Esurf + Ereac
	Eproduct  = Eprod_surf + Eproduct
	deltaE = Eproduct - Ereactant
	print("deltaE = %5.3e, Ereac = %5.3e, Eprod = %5.3e" % (deltaE, Ereactant, Eproduct))

	if check: view(surf)

	data = {"unique_id" : unique_id, "reaction_energy": deltaE}
	datum.append(data)

with open(outjson,"w") as f:
	json.dump(datum, f)
