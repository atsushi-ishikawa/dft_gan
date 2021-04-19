from ase import Atoms
from ase.build import add_adsorbate
from ase.visualize import view
from ase.calculators.emt import EMT
from ase.db import connect
from ase.optimize import BFGS
import json
import os
import pandas as pd
import time

surf_json = "surf.json"
reac_json = "reaction_energy.json"

# argvs   = sys.argv
# injson  = argvs[1] # json for surfaces
# outjson = argvs[2] # json for writing reaction energies

if not os.path.isfile(reac_json):
	with open(reac_json, "w") as f:
		f.write(json.dumps([], indent=4))
else:
	df_reac = pd.read_json(reac_json)
	df_reac = df_reac.set_index("unique_id")

db1  = connect(surf_json)
calc = EMT()

numdata = db1.count() + 1

check = False
#
# reactant
#
reac = Atoms("N2", [(0, 0, 0), (0, 0, 1.1)])
reac.set_calculator(calc)
print(" --- calculating %s ---" % reac.get_chemical_formula())
opt = BFGS(reac)
opt.run(fmax=0.1)
Ereac = reac.get_potential_energy()
#
# product
#
prod1 = Atoms("N", [(0, 0, 0)])
prod2 = Atoms("N", [(0, 0, 0)])
#
# loop over surfaces
#
for id in range(1, numdata):
	surf = db1.get_atoms(id=id)
	obj  = db1[id]
	data = obj.data
	unique_id = obj["unique_id"]

	try:
		#
		# search deltaE from existing file
		#
		deltaE = df_reac.loc[unique_id].reaction_energy
	except:
		#
		# not found
		#
		try:
			# check whether calculating or not
			status = df_reac.loc[unique_id]["status"]
			if status == "doing":
				print("somebody doing for %s" % unique_id)
				continue
		except:
			#
			# no record at all --- calculate here
			#
			data = {"unique_id": unique_id, "status": "doing"}
			#
			# updating and writing to json 
			#
			with open(reac_json, "r") as f:
				datum = json.load(f)
				datum.append(data)
			with open(reac_json, "w") as f:
				json.dump(datum, f, indent=4)

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
			Eproduct  = Eprod_surf
			deltaE = Eproduct - Ereactant
			print("deltaE = %5.3e, Ereac = %5.3e, Eprod = %5.3e" % (deltaE, Ereactant, Eproduct))

			if check: view(surf)
			#
			# add to database
			#
			data = {"unique_id": unique_id, "reaction_energy": deltaE, "status": "done"}
			with open(reac_json, "r") as f:
				datum = json.load(f)

				#  remove "doing" record as calculation is done
				for i in range(len(datum)):
					if datum[i]["unique_id"] == unique_id:
						datum.pop(i)
						break
				datum.append(data)
			with open(reac_json, "w") as f:
				json.dump(datum, f, indent=4)

