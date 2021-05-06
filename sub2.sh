#!/bin/sh

calculators=whisky
dir=/home/a_ishi/ase/nn_reac/

echo "downloading json files from $calculators:$dir"

# copy json file from VASP calculating clusters
scp $calculators:$dir/{surf,reaction_energy}.json ./

python nn_reac.py
python make_todo_list.py

scp ./{surf.json,todolist.txt} $calculators:$dir

