#!/bin/sh

cpu_team="whisky" # ("whisky | "vodka" | "mio" | "mio02")
dir=${HOME}/ase/nn_reac/
host=`hostname`

# copy json file from VASP calculating clusters
if [ $host != $cpu_team ]; then
	echo "downloading json files from $cpu_team:$dir"
	scp $cpu_team:$dir/{surf,reaction_energy}.json ./
fi
cp surf.json surf.json.bk
cp reaction_energy.json reaction_energy.json.bk

python rate.py
python energy_diagram.py

rm doing_reaction_energy_calc 2> /dev/null
touch doing_GAN

python nn_reac.py
python make_todo_list.py
#
# send
#
if [ $host != $cpu_team ]; then
	echo "sending json file and to-do-list to $cpu_team:$dir"
	scp ./{surf.json,todolist.txt} $cpu_team:$dir
fi

rm doing_GAN 2> /dev/null
touch doing_reaction_energy_calc

