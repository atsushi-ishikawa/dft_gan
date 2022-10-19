def read_reactionfile(file):
    import os
    import re

    f = open(file, "r")

    # drop comment and branck lines
    lines = f.readlines()
    newlines = []
    for line in lines:
        if not(re.match(r"^#", line)) and not(re.match(r"^s*$", line)):
            newlines.append(line)
    lines = newlines
    numlines = len(lines)

    reac = list(range(numlines))
    rxn  = list(range(numlines))
    prod = list(range(numlines))

    for i, line in enumerate(lines):
        text = line.replace("\n", "").replace(">", "").split("--")
        reac_tmp = text[0]
        rxn_tmp  = text[1]
        prod_tmp = text[2]

        reac[i] = re.split(" \+ ", reac_tmp)  # for cations
        prod[i] = re.split(" \+ ", prod_tmp)  # for cations

        reac[i] = remove_space(reac[i])
        prod[i] = remove_space(prod[i])

        rxn[i]  = reac[i][0] + "_" + rxn_tmp

    return reac, rxn, prod

def return_lines_of_reactionfile(file):
    import os
    import re

    # drop comment and branck lines
    f = open(file, "r")
    lines = f.readlines()
    newlines = []
    for line in lines:
        if not(re.match(r"^#", line)) and not(re.match(r"^s*$", line)):
            newlines.append(line)
    lines = newlines
    return lines

def remove_space(obj):
    newobj = [0]*len(obj)
    if isinstance(obj, str):
        #
        # string
        #
        newobj = obj.replace(" ", "")
    elif isinstance(obj, list):
        #
        # list
        #
        for i, obj2 in enumerate(obj):
            if isinstance(obj2, list):
                #
                # nested list
                #
                for ii, jj in enumerate(obj2):
                    jj = jj.strip()
                newobj[i] = jj
            elif isinstance(obj2, str):
                #
                # simple list
                #
                obj2 = obj2.replace(" ", "")
                newobj[i] = obj2
            elif isinstance(obj2, int):
                #
                # integer
                #
                newobj[i] = obj2
            else:
                newobj[i] = obj2
    else:  # error
        print("remove_space: input str or list")

    return newobj

def get_reac_and_prod(reactionfile):
    import numpy as np
    import os
    import sys

    # form reactant and product information
    (reac, rxn, prod) = read_reactionfile(reactionfile)

    rxn_num = len(rxn)

    r_ads  = list(range(rxn_num))
    r_site = [[] for _ in range(rxn_num)]
    r_coef = [[] for _ in range(rxn_num)]

    p_ads  = list(range(rxn_num))
    p_site = list(range(rxn_num))
    p_coef = list(range(rxn_num))

    for irxn, jrnx in enumerate(rxn):
        ireac = reac[irxn];     iprod = prod[irxn]
        ireac_num = len(ireac); iprod_num = len(iprod)
        #
        # reactant
        #
        r_ads[irxn]  = list(range(ireac_num))
        r_site[irxn] = list(range(ireac_num))
        r_coef[irxn] = list(range(ireac_num))

        for imol, mol in enumerate(ireac):
            r_site[irxn][imol] = []
            r_ads[irxn][imol]  = []
            #
            # coefficient
            #
            if "*" in mol:
                #r_coef[irxn][imol] = int(mol.split("*")[0])
                r_coef[irxn][imol] = float(mol.split("*")[0])
                rest = mol.split("*")[1]
            else:
                r_coef[irxn][imol] = 1
                rest = mol

            # site
            if ',' in rest:
                sites = rest.split(',')
                for isite, site in enumerate(sites):
                    r_site[irxn][imol].append(site.split('_')[1])
                    r_ads[irxn][imol].append(site.split('_')[0])
            elif '_' in rest:
                r_site[irxn][imol].append(rest.split('_')[1])
                r_ads[irxn][imol].append(rest.split('_')[0])
            else:
                r_site[irxn][imol].append('gas')
                r_ads[irxn][imol].append(rest)
        #
        # product
        #
        p_ads[irxn]  = list(range(iprod_num))
        p_site[irxn] = list(range(iprod_num))
        p_coef[irxn] = list(range(iprod_num))

        for imol, mol in enumerate(iprod):
            p_site[irxn][imol] = []
            p_ads[irxn][imol]  = []
            #
            # coefficient
            #
            if "*" in mol:
                #p_coef[irxn][imol] = int(mol.split("*")[0])
                p_coef[irxn][imol] = float(mol.split("*")[0])
                rest = mol.split("*")[1]
            else:
                p_coef[irxn][imol] = 1
                rest = mol

            # site
            if ',' in rest:
                sites = rest.split(',')
                for isite, site in enumerate(sites):
                    p_site[irxn][imol].append(site.split('_')[1])
                    p_ads[irxn][imol].append(site.split('_')[0])
            elif '_' in rest:
                p_site[irxn][imol].append(rest.split('_')[1])
                p_ads[irxn][imol].append(rest.split('_')[0])
            else:
                p_site[irxn][imol].append('gas')
                p_ads[irxn][imol].append(rest)

        # print("irxn=%d, %s-->%s, coef: %s-->%s, site:%s-->%s"
        #	% (irxn, r_ads[irxn], p_ads[irxn], r_coef[irxn], p_coef[irxn], r_site[irxn], p_site[irxn]))

    return (r_ads, r_site, r_coef,  p_ads, p_site, p_coef)


def get_number_of_reaction(reactionfile):
    import numpy as np
    import os
    import sys
    #
    # return number of elementary reactions
    #
    (reac, rxn, prod) = read_reactionfile(reactionfile)
    rxn_num = len(rxn)
    return rxn_num


def get_preexponential(reactionfile):
    #
    # ! not needed for MATLAB use
    #
    from ase import Atoms, Atom
    from ase.collections import methane
    import numpy as np
    import os
    import sys
    #
    # calculate pre-exponential factor
    #
    (r_ads, r_site, r_coef,  p_ads, p_site, p_coef) = get_reac_and_prod(reactionfile)

    rxn_num = get_number_of_reaction(reactionfile)

    Afor = np.array(rxn_num, dtype="f")
    Arev = np.array(rxn_num, dtype="f")

    mass_reac = np.array(rxn_num*[range(len(r_ads[0]))], dtype="f")
    mass_prod = np.array(rxn_num*[range(len(p_ads[0]))], dtype="f")

    for irxn in range(rxn_num):
        print("--------- %d ---------" % irxn)
        #
        # reactants
        #
        for imol, mol in enumerate(r_ads[irxn]):
            tmp  = methane[mol]
            mass = sum(tmp.get_masses())
            mass_reac[irxn, imol] = mass
        #
        # products
        #
        for imol, mol in enumerate(p_ads[irxn]):
            tmp  = methane[mol]
            mass = sum(tmp.get_masses())
            mass_prod[irxn, imol] = mass

        Afor[irxn] = 1.0
        Arev[irxn] = 1.0

    return Afor, Arev

def get_rateconstant(reactionfile, Afor, Arev, Efor, Erev, T):
    #
    # ! not needed for MATLAB use
    #
    from ase import Atoms, Atom
    from ase.collections import methane
    import numpy as np
    import os
    import sys
    #
    # calculate rate constant
    #
    # (r_ads, r_site, r_coef,  p_ads, p_site, p_coef) = get_reac_and_prod(reactionfile)

    rxn_num = get_number_of_reaction(reactionfile)

    kfor = np.array(rxn_num, dtype="f")
    krev = np.array(rxn_num, dtype="f")

    RT = 8.314*T / 1000.0  # in kJ/mol

    for irxn in range(rxn_num):
        kfor[irxn] = Afor[irxn]*exp(-Efor[irxn] / RT)
        krev[irxn] = Arev[irxn]*exp(-Erev[irxn] / RT)

    return kfor, krev


def read_speciesfile(speciesfile):

    f = open(speciesfile)

    species = f.read()

    species = species.replace('[', '')
    species = species.replace(']', '')
    species = species.replace(' ', '')
    species = species.replace('\n', '')
    species = species.replace('\'', '')
    species = species.split(",")

    return species

def remove_parentheses(file):
    import os
    #
    # remove parentheses -- maybe for MATLAB use
    #
    tmpfile = "ttt.txt"
    os.system('cat %s | sed "s/\[//g" > %s' % (file, tmpfile))
    os.system('cat %s | sed "s/\]//g" > %s' % (tmpfile, file))
    os.system('rm %s' % tmpfile)


def get_species_num(*species):
    # Return what is the number of species in speciesfile.
    # If argument is not present, returns the number of species.
    from reaction_tools import read_speciesfile

    speciesfile = "species.txt"
    lst = read_speciesfile(speciesfile)

    if len(species) == 0:
        # null argument: number of species
        return len(lst)
    else:
        # return species number
        spec = species[0]
        return lst.index(spec)

def get_adsorption_sites(infile):
    from reaction_tools import remove_space

    f = open(infile, "r")

    lines = f.readlines()

    mol  = list(range(len(lines)))
    site = list(range(len(lines)))

    for i, line in enumerate(lines):
        aaa, bbb = line.replace("\n", "").split(":")
        mol[i]  = remove_space(aaa)
        bbb     = remove_space(bbb)
        site[i] = bbb.split(",")

    return mol, site

def find_closest_atom(surf, offset=(0, 0)):
    from ase import Atoms, Atom
    from ase.build import add_adsorbate
    from ase.visualize import view
    import numpy as np

    dummy = Atom('H', (0, 0, 0))
    ads_height = 0.1
    add_adsorbate(surf, dummy, ads_height, position=(0, 0), offset=offset)
    natoms  = len(surf.get_atomic_numbers())
    last    = natoms-1
    # ads_pos = surf.get_positions(last)

    dist = surf.get_distances(last, [range(natoms)], vector=False)
    dist = np.array(dist)
    dist = np.delete(dist, last)  # delete adsorbate itself
    clothest = np.argmin(dist)

    return clothest

def sort_atoms_by_z(atoms):
    from ase import Atoms, Atom
    import numpy as np
    import collections
    #
    # keep information for original Atoms
    #
    tags = atoms.get_tags()
    pbc  = atoms.get_pbc()
    cell = atoms.get_cell()

    dtype = [("idx", int), ("z", float)]
    #
    # get set of chemical symbols
    #
    symbols  = atoms.get_chemical_symbols()
    elements = sorted(set(symbols), key=symbols.index)
    num_elem = []
    for i in elements:
        num_elem.append(symbols.count(i))

    #
    # loop over each groups
    #
    iatm = 0
    newatoms = Atoms()
    zcount = []
    for inum in num_elem:
        zlist  = np.array([], dtype=dtype)
        for idx in range(inum):
            tmp = np.array([(iatm, atoms[iatm].z)], dtype=dtype)
            zlist = np.append(zlist, tmp)
            iatm = iatm + 1

        zlist = np.sort(zlist, order="z")

        for i in zlist:
            idx = i[0]
            newatoms.append(atoms[idx])

        tmp = np.array([], dtype=float)
        for val in zlist:
            tmp = np.append(tmp, round(val[1], 2))
        tmp = collections.Counter(tmp)
        zcount.append(list(tmp.values()))
    #
    # restore tag, pbc, cell
    #
    newatoms.set_tags(tags)
    newatoms.set_pbc(pbc)
    newatoms.set_cell(cell)

    return newatoms, zcount

def get_number_of_valence_electrons(atoms):
    #
    # Returns number of valence electrons for VASP calculation.
    # Calls VASP calculation once.
    # Returned electron numbers should be ++1 or --1 for cations, anions, etc.
    #
    from ase.calculators.vasp import Vasp
    atoms.calc = Vasp(prec="normal", xc="PBE", encut=300.0, nsw=0, nelm=1)
    nelec = atoms.calc.get_number_of_electrons()
    return nelec

def read_charge(mol):
    charge = 0.0  # initial
    if "^" in mol:
        neutral = False
        mol, charge = mol.split('^')
        charge = float(charge.replace('{', '').replace('}', '').replace('+', ''))
    else:
        neutral = True

    return mol, neutral, charge

def remove_side_and_flip(mol):
    #
    # remove SIDE and FLIP in molecule
    #
    if '-SIDEx' in mol:
        mol = mol.replace('-SIDEx', '')
    elif '-SIDEy' in mol:
        mol = mol.replace('-SIDEy', '')
    elif '-SIDE' in mol:
        mol = mol.replace('-SIDE', '')
    elif '-FLIP' in mol:
        mol = mol.replace('-FLIP', '')
    elif '-TILT' in mol:
        mol = mol.replace('-TILT', '')
    elif '-HIGH' in mol:
        mol = mol.replace('-HIGH', '')

    return mol

def neb_copy_contcar_to_poscar(nimages):
    #
    # copy 0X/CONTCAR to 0X/POSCAR after NEB run
    #
    import os
    for images in range(nimages):
        os.system('cp %02d/CONTCAR %02d/POSCAR' % (images+1, images+1))


def make_it_closer_by_exchange(atom1, atom2, thre=0.05):
    # thre: when distance is larger than this value, do switch
    from ase import Atoms, Atom
    from ase.geometry import distance

    natoms = len(atom1)
    const_list = atom1.constraints[0].get_indices()
    # Constrained atoms. Do not exchange these.

    for i in range(natoms):
        for j in range(i+1, natoms):
            if atom1[i].symbol == atom1[j].symbol:
                # measure distance between "ATOMS" (in ASE object)
                dis_bef = distance(atom1, atom2)
                atom1B  = atom1.copy()
                atom1B.positions[[i, j]] = atom1B.positions[[j, i]]
                dis_aft = distance(atom1B, atom2)

                if (dis_aft + thre < dis_bef) and not (i in const_list) and not (j in const_list):
                    #
                    # switch
                    #
                    print("exchanged {0:3d} and {1:3d}: (dis_bef, dis_aft) = ({2:6.2f},{3:6.2f})".
                          format(i, j, dis_bef, dis_aft))
                    tmp = atom1B.numbers[i]
                    atom1B.numbers[i] = atom1B.numbers[j]
                    atom1B.numbers[j] = tmp

                    atom1 = atom1B.copy()

    return atom1
