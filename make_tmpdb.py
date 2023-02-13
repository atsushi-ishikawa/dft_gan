from ase.db import connect
from ase import Atoms
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--tmpdb_name", type=str, default="tmp.db")
args = parser.parse_args()
tmpdb = args.tmpdb_name

he = Atoms("He")
db = connect(tmpdb)
db.write(he)

