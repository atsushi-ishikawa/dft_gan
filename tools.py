def ABcoord(mol,A,B):
	from ase import Atoms
	import numpy as np

	symbols = np.array( mol.get_chemical_symbols() )
	A_idx   = np.where(symbols==A)[0]
	B_list  = np.where(symbols==B)[0]
	AB_dist = mol.get_distances(A_idx, B_list)

	R_AB = np.min(AB_dist)
	coordinatingB = B_list[np.argmin(AB_dist)]

	return R_AB,coordinatingB


def run_packmol(xyz_file, a, num, outfile):
	import os

	packmol = "/Users/ishi/packmol/packmol"
	filetype = "xyz"

	cell1 = [0.0, 0.0, 0.0, a, a, a]
	cell2 = " ".join(map(str, cell1))

	f = open("pack_tmp.inp", "w")
	text = [
		"tolerance 2.0"             + "\n",
		"output "     + outfile     + "\n",
		"filetype "   + filetype    + "\n",
		"structure "  + xyz_file    + "\n",
		"  number "   + str(num)    + "\n",
		"  inside box " + cell2     + "\n",
		"end structure"
		]
	f.writelines(text)
	f.close()

	run_string = packmol + " < pack_tmp.inp"

	os.system(run_string)

	# os.system("rm pack_tmp.inp")


def json_to_csv(jsonfile, csvfile):
	import json
	import pandas as pd
	from pandas.io.json import json_normalize
	f = open(jsonfile, "r")
	d = json.load(f)

	dd = []
	nrec = len(d)
	for i in range(1, nrec):
		if str(i) in d:
			tmp = d[str(i)]
			dd.append(json_normalize(tmp))

	ddd = pd.concat(dd)

	newcol = []
	for key in ddd.columns:
		key = key.replace("calculator_parameters.", "")
		key = key.replace("key_value_pairs.", "")
		key = key.replace("data.", "")
		newcol.append(key)

	ddd.columns = newcol

	# sort data by "num"
	if "num" in ddd.columns:
		ddd2 = ddd.set_index("num")
		ddd  = ddd2.sort_index()

	ddd.to_csv(csvfile)


def load_ase_json(jsonfile):
	import json
	import pandas as pd
	f = open(jsonfile, "r")
	d = json.load(f)

	dd = []
	nrec = len(d)
	for i in range(1, nrec):
		if str(i) in d:
			tmp = d[str(i)]
			dd.append(pd.json_normalize(tmp))

	ddd = pd.concat(dd)

	newcol = []
	for key in ddd.columns:
		key = key.replace("calculator_parameters.", "")
		key = key.replace("key_value_pairs.", "")
		key = key.replace("data.", "")
		newcol.append(key)

	ddd.columns = newcol

	# sort data by "num"
	if "num" in ddd.columns:
 		ddd2 = ddd.set_index("num")
 		ddd  = ddd2.sort_index()

	return ddd


def delete_num_from_json(num,jsonfile):
	from ase.db import connect
	import sys

	db = connect(jsonfile)
	id = db.get(num=num).id
	db.delete([id])


def sort_atoms_by(atoms, xyz="x"):
	from ase import Atoms, Atom
	import numpy as np
	#
	# keep information for original Atoms
	#
	tags = atoms.get_tags()
	pbc  = atoms.get_pbc()
	cell = atoms.get_cell()
	dtype = [("idx", int), (xyz, float)]
	list = np.array([], dtype=dtype)


	for idx, atom in enumerate(atoms):
		if xyz == "x":
			tmp = np.array([(idx, atom.x)], dtype=dtype)
		elif xyz == "y":
			tmp = np.array([(idx, atom.y)], dtype=dtype)
		else:
			tmp = np.array([(idx, atom.z)], dtype=dtype)

		list = np.append(list, tmp)

	list = np.sort(list, order=xyz)

	newatoms = Atoms()
	for i in list:
		idx = i[0]
		newatoms.append(atoms[idx])

	# restore
	newatoms.set_tags(tags)
	newatoms.set_pbc(pbc)
	newatoms.set_cell(cell)

	return newatoms


#def fix_lower_surface(atoms, nlayer, nrelax):
#    import numpy as np
#    from ase.constraints import FixAtoms
#
#    newatoms = sort_atoms_by_z(atoms)
#    natoms   = len(newatoms.get_atomic_numbers())
#    one_surf = natoms // nlayer
#    tag = np.ones(natoms, int)
#    for i in range(natoms-1, natoms-nrelax*one_surf-1, -1):
#        tag[i] = 0
#    newatoms.set_tags(tag)
#    c = FixAtoms(indices=[atom.index for atom in newatoms if atom.tag == 1])
#    newatoms.set_constraint(c)
#    return newatoms


def get_number_of_layers(atoms):
	import numpy as np

	pos    = atoms.positions
	zpos   = np.round(pos[:,2], decimals=5)
	nlayer = len(list(set(zpos)))

	return nlayer


def set_tags_by_z(atoms):
	import numpy as np
	import pandas as pd

	newatoms = atoms.copy()
	pos   = newatoms.positions
	zpos  = np.round(pos[:,2], decimals=1)
	bins  = list(set(zpos))
	bins  = np.array(bins) + 1.0e-2
	bins  = np.insert(bins,0,0)

	labels = []
	for i in range(len(bins)-1):
		labels.append(i)

	tags = pd.cut(zpos, bins=bins, labels=labels).to_list()
	newatoms.set_tags(tags)
	
	return newatoms


def fix_lower_surface(atoms):
	import numpy as np
	import pandas as pd
	from ase.constraints import FixAtoms

	newatoms = atoms.copy()

	# set tags
	newatoms = set_tags_by_z(newatoms)
	tags  = newatoms.get_tags()

	# constraint
	nlayer = get_number_of_layers(newatoms)
	lower  = list(range(nlayer // 2))
	c = FixAtoms(indices=[iatom.index for iatom in newatoms if iatom.tag in lower])
	newatoms.set_constraint(c)

	return newatoms


def find_highest(json, score):
	import pandas as pd

	df = pd.read_json(json)
	df = df.set_index("unique_id")
	df = df.dropna(subset=[score])
	df = df.sort_values(score, ascending=False)
	
	best = df.iloc[0].name

	return best


def make_step(atoms):
	import numpy as np
	from ase.build import rotate

	newatoms = atoms.copy()
	newatoms = sort_atoms_by(newatoms, xyz="z")

	nlayer    = get_number_of_layers(newatoms)
	perlayer  = len(newatoms) // nlayer
	toplayer  = newatoms[-perlayer:]
	top_layer = sort_atoms_by(toplayer, xyz="y")

	# first remove top layer then add sorted top layer
	del newatoms[-perlayer:]
	newatoms += top_layer

	remove = perlayer // 2

	nstart = perlayer*(nlayer-1)  # index for the atom starting the top layer
	del newatoms[nstart:nstart+remove]

	return newatoms


def mirror_invert(atoms):
	import numpy as np

	pos  = atoms.get_positions()
	cell = atoms.cell

	# set position
	pos[:,0] = -pos[:,0]
	atoms.set_positions(pos)

	# set cell
	cell = [[-cell[i][0], cell[i][1], cell[i][2]] for i in range(3)]
	cell = np.array(cell)
	cell = np.round(cell + 1.0e-5, decimals=4)
	atoms.set_cell(cell)

	return atoms

