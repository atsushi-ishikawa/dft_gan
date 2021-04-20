from ase.build import fcc111
from ase.visualize import view
from ase.calculators.emt import EMT
from ase.db import connect
import numpy as np
import os
import sys
import random

# lattice constant
a = {"Rh": 3.80}

surf  = fcc111(symbol="Rh", size=[3, 3, 4], a=a["Rh"], vacuum=10.0)
check = False # check structure or not
calc  = EMT()
#
# replace atoms by some element in the list
#
natoms = len(surf.get_atomic_numbers())
max_replace = int(0.3 * natoms)  # 5
# elementlist = ["Al", "Cu", "Ag", "Au", "Ni", "Pt"]
elementlist = ["Ir"]

outjson = "surf.json"

# remove old one
if os.path.exists(outjson):
	os.remove(outjson)

argvs = sys.argv
num_data = int(argvs[1])

db = connect(outjson)
#
# shuffle
#
id = 1
for i in range(num_data):
	#
	# make shuffled surface
	#
	surf_copy = surf.copy()

	# how many atoms to replace?
	num_replace = random.choice(range(1, max_replace + 1))

	data = {}
	# replace element
	for iatom in range(num_replace):
		surf_copy[iatom].symbol = random.choice(elementlist)

	# get element atomic_numbers
	atomic_numbers = surf_copy.get_atomic_numbers()
	# shuffle element atomic_numbers
	np.random.shuffle(atomic_numbers)
	# set new element atomic_numbers
	surf_copy.set_atomic_numbers(atomic_numbers)
	formula = surf_copy.get_chemical_formula()

	surf_copy.set_calculator(calc)

	if check: view(surf_copy)

<<<<<<< HEAD:make_surf.py
	data = {"chemical_formula": formula, "atomic_numbers": atomic_numbers, "run": 0}
	db.write(surf_copy, data=data)
=======
    data = {"chemical_formula": formula, "atomic_numbers": atomic_numbers, "run": 0}
    db.write(surf_copy, data=data)
    db.update(id, status="reaction_energy_not_done")
    id += 1
>>>>>>> whisky:make_surf2.py
