from ase import Atoms, Atom
from ase.build import fcc111
from ase.visualize import view
from ase.calculators.emt import EMT
from ase.db import connect
import numpy as np
import os, sys
import random

surf = fcc111(symbol="Pd", size=[4,4,4], a=4.0, vacuum=10.0)
calc = EMT()
#
# replace atoms by some element in the list
#
max_replace = 5
#elementlist = ["Al", "Cu", "Ag", "Au", "Ni", "Pt"]
elementlist = ["Pt"]

outjson = "surf.json"

# remove old one
if os.path.exists(outjson):
	os.remove(outjson)

datum = []
argvs = sys.argv
num_data = int(argvs[1])

db = connect(outjson)

check = False # check structure or not
#
# shuffle
#
for i in range(num_data):
	#
	# make shuffled surface
	#
	surf_copy = surf.copy()

	# how many atoms to replace?
	num_replace = random.choice(range(1,max_replace+1))

	data = {}
	# replace element
	for iatom in range(num_replace):
		surf_copy[iatom].symbol = random.choice(elementlist)

	# get element index
	index = surf_copy.get_atomic_numbers()
	# shuffle element index
	np.random.shuffle(index)
	# set new element index
	surf_copy.set_atomic_numbers(index)
	formula = surf_copy.get_chemical_formula()

	surf_copy.set_calculator(calc)

	if check: view(surf_copy)

	data = {"chemical_formula" : formula, "atomic_numbers" : index}
	datum.append(data)
	db.write(surf_copy, data=data)

