from ase import Atoms
from ase.build import add_adsorbate
from ase.visualize import view
from ase.calculators.vasp import Vasp
from ase.calculators.emt import EMT
from ase.db import connect
from ase.optimize import BFGS
from tools import fix_lower_surface
import sys
import json
import os
import pandas as pd
import numpy as np
import argparse
import shutil

clean = False

parser = argparse.ArgumentParser()
parser.add_argument("--calculator", default="emt", choices=["emt", "EMT", "vasp", "VASP"])
parser.add_argument("--surf_json",  default="surf.json", help="json for surfaces")
parser.add_argument("--reac_json",  default="reaction_energy.json", help="json for writing reaction energies")

args = parser.parse_args()
calculator = args.calculator.lower()
surf_json  = args.surf_json
reac_json  = args.reac_json

# remove old one
# if os.path.exists(outjson):
#	os.remove(outjson)

if not os.path.isfile(reac_json):
    # reac_json does not exist -- make
    with open(reac_json, "w") as f:
        f.write(json.dumps([], indent=4))
else:
    try:
        df_reac = pd.read_json(reac_json)
        df_reac = df_reac.set_index("unique_id")
    except:
        pass

db1 = connect(surf_json)
steps = 50  # maximum number of geomtry optimization steps

if "vasp" in calculator:
    prec = "normal"
    xc   = "beef-vdw"
    ivdw = 0
    nsw  = 0
    nelm   = 50
    ibrion = -1
    algo = "VeryFast"
    kpts_mol  = [1, 1, 1]
    kpts_surf = [3, 3, 1]
    kgamma = True
    pp    = "potpaw_PBE.54"
    npar  = 12
    nsim  = npar
    kpar  = 1
    isym  = 0
    lreal = True

    calc_mol  = Vasp(label=None, prec=prec, xc=xc, ivdw=ivdw, algo=algo, ibrion=ibrion ,nsw=nsw, nelm=nelm,
                     kpts=kpts_mol,  kgamma=kgamma, pp=pp, npar=npar, nsim=nsim, kpar=kpar, isym=isym, lreal=lreal)
    calc_surf = Vasp(label=None, prec=prec, xc=xc, ivdw=ivdw, algo=algo, ibrion=ibrion ,nsw=nsw, nelm=nelm,
                     kpts=kpts_surf, kgamma=kgamma, pp=pp, npar=npar, nsim=nsim, kpar=kpar, isym=isym, lreal=lreal)
else:
    calc_mol  = EMT()
    calc_surf = EMT()

numdata = db1.count() + 1

check = False

def set_unitcell(Atoms, vacuum=10.0):
	import numpy as np
	cell = np.array([1, 1, 1])*vacuum
	Atoms.set_cell(cell)

def set_calculator_with_label(Atoms, calc, label=None):
    if "vasp" in calculator:
        if label is None:
            name = Atoms.get_chemical_formula()
            calc.set_label(name)
        else:
            calc.set_label(label)
    Atoms.set_calculator(calc)


def run_optimizer(atoms, steps=10):
    fmax  = 0.1

    calc = atoms.get_calculator()
    if calc.name.lower() == "emt":
        opt = BFGS(atoms)
        opt.run(fmax=fmax, steps=steps)
    elif calc.name.lower() == "vasp":
        calc.int_params["ibrion"] = 2
        calc.int_params["nsw"] = steps
        calc.input_params["potim"] = 0.1
        atoms.set_calculator(calc)
        atoms.get_potential_energy()
    else:
        print("use vasp or emt. now ", calc.name)
        sys.exit()

    print(" ------- geometry optimization finished ------- ")
	#
    # reset vasp calculator to single point energy's one
	#
    if calc.name.lower() == "vasp":
        calc.int_params["ibrion"] = -1
        calc.int_params["nsw"] = 0
        atoms.set_calculator(calc)

#
# loop over surfaces
#
for id in range(1, numdata):
    surf = db1.get_atoms(id=id)
    obj  = db1[id]
    data = obj.data
    unique_id = obj["unique_id"]
    status = obj.status

    try:
        #
        # search for old file
        #
        if status == "calculating_reaction_energy":
            print("other node calculating this one. skip")
            continue
        else:
            deltaE = df_reac.loc[unique_id].reaction_energy
    except:
        #
        # not found -- calculate here
        #
        print(" --- calculating %s ---" % surf.get_chemical_formula())
        db1.update(id, status="calculating_reaction_energy")

        #
        # reactant
        #
        reac = Atoms("N2", [(0, 0, 0), (0, 0, 1.1)])
        set_unitcell(reac)
        label = reac.get_chemical_formula() + "_" + unique_id
        set_calculator_with_label(reac, calc_mol, label=label)
        run_optimizer(reac, steps=steps)
        Ereac = reac.get_potential_energy()
        if clean: shutil.rmtree(label)
        #
        # product
        #
        prod1 = Atoms("N", [(0, 0, 0)])
        prod2 = Atoms("N", [(0, 0, 0)])

        #
        # surface
        #
        nlayer = 4
        nrelax = nlayer // 2
        surf = fix_lower_surface(surf, nlayer, nrelax)

        label = surf.get_chemical_formula() + "_" + unique_id
        set_calculator_with_label(surf, calc_surf, label=label)
        run_optimizer(surf, steps=steps)
        Esurf = surf.get_potential_energy()
        if clean: shutil.rmtree(label)
        #
        # surface + adsorbate
        #
        add_adsorbate(surf, prod1, offset=(0.3, 0.3), height=1.3)
        add_adsorbate(surf, prod2, offset=(0.6, 0.6), height=1.3)
        #
        print(" --- calculating %s ---" % surf.get_chemical_formula())
        #
        label = surf.get_chemical_formula() + "_" + unique_id
        set_calculator_with_label(surf, calc_surf, label=label)
        run_optimizer(surf, steps=steps)
        Eprod_surf = surf.get_potential_energy()
        if clean: shutil.rmtree(label)

        Ereactant = Esurf + Ereac
        Eproduct  = Eprod_surf
        deltaE = Eproduct - Ereactant
        print("deltaE = %5.3e, Ereac = %5.3e, Eprod = %5.3e" % (deltaE, Ereactant, Eproduct))

        if check: view(surf)

        data = {"unique_id": unique_id, "reaction_energy": deltaE}
        with open(reac_json) as f:
            datum = json.load(f)
            datum.append(data)
        #
        # add to database
        #
        with open(reac_json, "w") as f:
            json.dump(datum, f, indent=4)

        db1.update(id, status="reaction_energy_done")
