#!/bin/bash

cp surf_bk.json surf.json ; cp reaction_energy_bk.json reaction_energy.json ; rm ./log/* 2> /dev/null

max=9
for (( i=1 ; i<=$max; i++ )) ; do
	echo "--------------- run = " $i "/" $max "---------------"
	python nn_torch.py && python calc_reaction_energy.py && python plot.py
done
