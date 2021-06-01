from ase.build import fcc111
from ase.visualize import view
from ase.calculators.emt import EMT
from ase.db import connect
import numpy as np
import os
import sys
import random

# lattice constant
#elem = {"symbol": "Pt", "a": 3.90}
elem = {"symbol": "Ni", "a": 3.52}

#surf  = fcc111(symbol=elem["symbol"], size=[2, 2, 5], a=elem["a"], vacuum=10.0)
surf  = fcc111(symbol=elem["symbol"], size=[3, 3, 4], a=elem["a"], vacuum=10.0)
surf.pbc = True
check = False # check structure or not
calc  = EMT()
#
# replace atoms by some element in the list
#
natoms = len(surf.get_atomic_numbers())
max_replace = int(0.3 * natoms)
# elementlist = ["Al", "Cu", "Ag", "Au", "Ni", "Pt"]
#elem2 = ["Rh"]
#elem2 = ["Pd"]
elem2 = ["Ru"]

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
		surf_copy[iatom].symbol = random.choice(elem2)

	# get element atomic_numbers
	atomic_numbers = surf_copy.get_atomic_numbers()
	atomic_numbers = list(atomic_numbers)  # make non-numpy

	# shuffle element atomic_numbers
	np.random.shuffle(atomic_numbers)

	# set new element atomic_numbers
	surf_copy.set_atomic_numbers(atomic_numbers)
	formula = surf_copy.get_chemical_formula()

	surf_copy.set_calculator(calc)

	if check: view(surf_copy)

	data = {"chemical_formula": formula, "atomic_numbers": atomic_numbers, "run": 0}
	db.write(surf_copy, data=data)
	id += 1

