from ase import Atoms
import numpy as np


def ABcoord(mol, A, B):
    symbols = np.array(mol.get_chemical_symbols())
    A_idx   = np.where(symbols == A)[0]
    B_list  = np.where(symbols == B)[0]
    AB_dist = mol.get_distances(A_idx, B_list)

    R_AB = np.min(AB_dist)
    coordinatingB = B_list[np.argmin(AB_dist)]

    return R_AB, coordinatingB


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


def delete_num_from_json(num, jsonfile):
    from ase.db import connect
    import sys

    db = connect(jsonfile)
    id = db.get(num=num).id
    db.delete([id])


def sort_atoms_by(atoms, xyz="x"):
    # keep information for original Atoms
    tags = atoms.get_tags()
    pbc  = atoms.get_pbc()
    cell = atoms.get_cell()
    dtype = [("idx", int), (xyz, float)]

    newatoms = Atoms()
    symbols = list(set(atoms.get_chemical_symbols()))
    for symbol in symbols:
        subatoms = Atoms(list(filter(lambda x: x.symbol == symbol, atoms)))
        atomlist = np.array([], dtype=dtype)
        for idx, atom in enumerate(subatoms):
            if xyz == "x":
                tmp = np.array([(idx, atom.x)], dtype=dtype)
            elif xyz == "y":
                tmp = np.array([(idx, atom.y)], dtype=dtype)
            else:
                tmp = np.array([(idx, atom.z)], dtype=dtype)

            atomlist = np.append(atomlist, tmp)

        atomlist = np.sort(atomlist, order=xyz)

        for i in atomlist:
            idx = i[0]
            newatoms.append(subatoms[idx])

    # restore
    newatoms.set_tags(tags)
    newatoms.set_pbc(pbc)
    newatoms.set_cell(cell)

    return newatoms


def get_number_of_layers(atoms):
    symbols = list(set(atoms.get_chemical_symbols()))
    symbols = sorted(symbols)
    nlayers = []

    for symbol in symbols:
        subatoms = Atoms(list(filter(lambda x: x.symbol == symbol, atoms)))
        pos  = subatoms.positions
        zpos = np.round(pos[:, 2], decimals=4)
        nlayers.append(len(list(set(zpos))))

    return nlayers


def set_tags_by_z(atoms):
    import pandas as pd

    pbc  = atoms.get_pbc()
    cell = atoms.get_cell()

    newatoms = Atoms()
    symbols = list(set(atoms.get_chemical_symbols()))
    symbols = sorted(symbols)

    for symbol in symbols:
        subatoms = Atoms(list(filter(lambda x: x.symbol == symbol, atoms)))
        pos  = subatoms.positions
        zpos = np.round(pos[:, 2], decimals=1)
        bins = list(set(zpos))
        bins = np.sort(bins)
        bins = np.array(bins) + 1.0e-2
        bins = np.insert(bins, 0, 0)

        labels = []
        for i in range(len(bins)-1):
            labels.append(i)

        tags = pd.cut(zpos, bins=bins, labels=labels).to_list()

        subatoms.set_tags(tags)
        newatoms += subatoms

    # restore
    newatoms.set_pbc(pbc)
    newatoms.set_cell(cell)

    return newatoms

def remove_layer(atoms=None, symbol=None, higher=1):
    import pandas as pd
    from ase.constraints import FixAtoms

    pbc  = atoms.get_pbc()
    cell = atoms.get_cell()

    atoms_copy = atoms.copy()

    # sort
    atoms_copy = sort_atoms_by(atoms_copy, xyz="z")

    # set tags
    atoms_copy = set_tags_by_z(atoms_copy)

    newatoms = Atoms()

    tags = atoms_copy.get_tags()
    maxtag = max(list(tags))

    for i, atom in enumerate(atoms_copy):
        if atom.tag >= maxtag - higher + 1 and atom.symbol == symbol:
            # remove this atom
            pass
        else:
            newatoms += atom

    newatoms.set_pbc(pbc)
    newatoms.set_cell(cell)

    return newatoms

def fix_lower_surface(atoms):
    import pandas as pd
    from ase.constraints import FixAtoms
    from ase.visualize import view

    newatoms = atoms.copy()

    # sort
    newatoms = sort_atoms_by(newatoms, xyz="z")

    # set tags
    newatoms = set_tags_by_z(newatoms)

    ### constraint

    # prepare symbol dict
    symbols_ = list(set(atoms.get_chemical_symbols()))
    symbols_ = sorted(symbols_)
    symbols = {}
    for i, sym in enumerate(symbols_):
        symbols.update({sym: i})

    # Determine fixlayer, which is a list of elements. Half of nlayers.
    nlayers = get_number_of_layers(newatoms)
    div = [i // 2 for i in nlayers]
    mod = [i % 2 for i in nlayers]
    fixlayers = [i+j for (i, j) in zip(div, mod)]

    # list of fixed atoms
    fixlist = []
    tags = newatoms.get_tags()
    minind = np.argmin(tags)
    maxind = np.argmax(tags)
    lowest_z  = newatoms[minind].position[2]
    highest_z = newatoms[maxind].position[2]
    z_thre = (highest_z - lowest_z) / 2

    for iatom in newatoms:
        ind = symbols[iatom.symbol]
        z_pos = iatom.position[2]
        if iatom.tag < fixlayers[ind] and z_pos < z_thre:
            fixlist.append(iatom.index)
        else:
            pass

    c = FixAtoms(indices=fixlist)

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


def mirror_invert(atoms, direction="x"):
    pos  = atoms.get_positions()
    cell = atoms.cell

    # set position
    pos[:, 0] = -pos[:, 0]
    atoms.set_positions(pos)

    # set cell
    if direction == "x":
        cell = [[-cell[i][0], cell[i][1], cell[i][2]] for i in range(3)]

    cell = np.array(cell)
    cell = np.round(cell + 1.0e-5, decimals=4)
    atoms.set_cell(cell)

    return atoms
