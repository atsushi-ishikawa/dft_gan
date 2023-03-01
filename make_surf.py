from ase.build import fcc111, fcc211, hcp0001, bulk, surface
from ase.visualize import view
from ase.calculators.emt import EMT
from ase.db import connect
from ase.io import read, write
from tools import make_step, remove_layer
import numpy as np
import os
import sys
import random
import argparse

parser = argparse.ArgumentParser()

args_list = lambda x: list(map(str, x.replace("[", "").replace("]", "").replace('"', '').split(",")))  # allow list for elements

parser.add_argument("--num_data", default=1, type=int, help="number of surfaces generating")
parser.add_argument("--check", action="store_true", help="check structure or not")
parser.add_argument("--surf_geom", default="fcc111", choices=["fcc111", "step_fcc", "step_hcp"])
parser.add_argument("--vacuum", default=10.0, type=float, help="length of vacuum layer")
parser.add_argument("--elem_from", type=args_list, default="Ru", help='elements to be replaced (e.g. ["Ru", "Ir"]')
parser.add_argument("--elem_to", type=args_list, default="Rh", help='elements to replace (e.g. ["Ru", "Ir"]')
parser.add_argument("--max_replace_percent", default=100, type=int, help="max percent of second element")
parser.add_argument("--replace_pattern", type=str, default="random", help="how to do atom replacement (random or all); num_data becomes #elem_to")
parser.add_argument("--cif", default=None)

args = parser.parse_args()
outjson = "surf.json"

cif = args.cif
num_data  = args.num_data
check     = args.check
surf_geom = args.surf_geom
vacuum    = args.vacuum

elem_from = args.elem_from
elem_to   = args.elem_to
replace_pattern = args.replace_pattern

max_rep = float(args.max_replace_percent)

print("making base surface ... result will be stored on {}".format(outjson))

# load base cif
if cif is not None:
    bulk = read(cif)
    surf = surface(bulk, indices=[1, 1, 0], layers=4, vacuum=vacuum, periodic=True)
    surf = surf*[1, 2, 1]
else:
    # lattice constant
    element  = "Pt"  # default
    lattice_const = {"Ru": 2.7*1.4, "Pt": 3.9, "Ni": 3.5}
    elem = {"symbol": element, "a": lattice_const[element]}

    if surf_geom == "fcc111":
        nlayer = 3
        surf = fcc111(symbol=elem["symbol"], size=[4, 4, nlayer], a=elem["a"], vacuum=vacuum,
                      orthogonal=True, periodic=True)
    elif surf_geom == "step_fcc":
        nlayer = 4
        surf = fcc211(symbol=elem["symbol"], size=[6, 3, nlayer], a=elem["a"], vacuum=vacuum,
                      orthogonal=True, periodic=True)
    elif surf_geom == "step_hcp":
        nlayer = 4
        surf = hcp0001(symbol=elem["symbol"], size=[4, 6, nlayer], a=elem["a"], vacuum=vacuum,
                       orthogonal=True, periodic=True)
        surf = make_step(surf)

## stepped non fcc
#bulk = bulk(elem["symbol"], "fcc", a=elem["a"], cubic=True, orthorhombic=False)
#bulk = bulk(elem["symbol"], "hcp", a=elem["a"], cubic=False, orthorhombic=True)
#surf = surface(bulk, indices=[1,1,1], layers=9, vacuum=vacuum)
#surf = surf*[3,3,1]

surf.translate([0, 0, -vacuum+0.1])
surf.wrap()

calc  = EMT()

# fake replacement of Ru --> Pt to use EMT
#for iatom in surf:
#    if iatom.symbol == "Ru":
#        iatom.symbol = "Pt"

# remove old one
if os.path.exists(outjson):
    os.remove(outjson)

db = connect(outjson)
#
# replace atoms by some element in the list
#
natoms = len(list(filter(lambda x: x in elem_from, surf.get_chemical_symbols())))
max_replace = int((max_rep/100)*natoms)

random.seed(111)
data = {}
oldlist = []
id = 1
counter = 0
#for i in range(num_data):
while counter < num_data:
    surf_copy = surf.copy()
    surf_copy = remove_layer(atoms=surf_copy, symbol="O", higher=1)
    #
    # make replaced surface
    #

    # how many atoms to replace?
    if max_replace != 0:
        num_replace = random.choice(range(0, max_replace + 1))
    else:
        num_replace = 0

    ## replace element
    # all 
    if replace_pattern == "all":
        replacing = elem_to.pop()
        for iatom in surf_copy:
            if iatom.symbol in elem_from:
                iatom.symbol = replacing
    # random
    else:
        from_list = []
        for iatom in surf_copy:
            if iatom.symbol != "O":
                from_list.append(iatom.index)

        from_list = random.sample(from_list, num_replace)

        for iatom in from_list:
            surf_copy[iatom].symbol = random.choice(elem_to)

    # get element atomic_numbers
    atomic_numbers = surf_copy.get_atomic_numbers()
    atomic_numbers = list(atomic_numbers)  # make non-numpy
    
    formula = surf_copy.get_chemical_formula()

    surf_copy.set_calculator(calc)

    if check: view(surf_copy)

    if atomic_numbers in oldlist:
        # found overlap in the list ... go next
        pass
    else:
        data = {"chemical_formula": formula, "atomic_numbers": atomic_numbers, "run": 0}
        db.write(surf_copy, data=data)
        write("POSCAR", surf_copy)
        oldlist.append(atomic_numbers)
        counter += 1
        id += 1
