from ase.build import fcc111, fcc211, bulk, surface
from ase.visualize import view
from ase.calculators.emt import EMT
from ase.db import connect
from ase.io import read, write
import numpy as np
import os
import sys
import random

argvs = sys.argv
num_data = int(argvs[1])

vacuum = 9.0

# lattice constant
#elem = {"symbol": "Pt", "a": 3.9}
#elem = {"symbol": "Ni", "a": 3.5}
elem = {"symbol": "Ru", "a": 2.7*1.3}

## flat
#surf = fcc111(symbol=elem["symbol"], size=[2, 2, 5], a=elem["a"], vacuum=vacuum)
#surf = fcc111(symbol=elem["symbol"], size=[3, 3, 4], a=elem["a"], vacuum=vacuum)

## stepped - fcc
#surf = fcc211(symbol=elem["symbol"], size=[6, 4, 4], a=elem["a"], vacuum=vacuum)
surf = fcc211(symbol=elem["symbol"], size=[6, 3, 4], a=elem["a"], vacuum=vacuum)

## stepped - non fcc
#bulk = bulk(elem["symbol"], "fcc", a=elem["a"], cubic=True, orthorhombic=False)
#surf = surface(bulk, indices=[5,1,1], layers=9, vacuum=vacuum)
#surf = surf*[1,2,1]

surf.pbc = True
surf.translate([0, 0, -vacuum+1.0])

check = False # check structure or not
calc  = EMT()
#
# replace atoms by some element in the list
#
natoms = len(surf.get_atomic_numbers())
max_replace = int(0.2* natoms)
#elem2 = ["Rh"]
elem2 = ["Pd"]
#elem2 = ["Ru"]

outjson = "surf.json"

# remove old one
if os.path.exists(outjson):
	os.remove(outjson)


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
	#write("POSCAR", surf_copy)
	id += 1

