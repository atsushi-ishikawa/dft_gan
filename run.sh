#!/bin/bash

datanum=200
dataset="old" # "new" or "old"
overwrite_bk="False"
loss_file="loss.h5"

# remove progress file
rm doing_* 2> /dev/null

# refresh loss file
rm $loss_file 2> /dev/null
python make_h5.py $loss_file

if [ $dataset = "new" ]; then
	rm surf.json reaction_energy.json 2> /dev/null
	python make_surf.py $datanum
	touch "doing_preparation"
	python calc_reaction_energy.py
	rm "doing_preparation"
	#
	# keep json files before GAN loop
	#
	cp surf.json surf_${datanum}_bk.json
	cp reaction_energy.json reaction_energy_${datanum}_bk.json
elif [ $dataset = "old" ]; then
	cp surf_${datanum}_bk.json surf.json
	cp reaction_energy_${datanum}_bk.json reaction_energy.json
else
	exit
fi

rm ./log/* 2> /dev/null

max=10
for (( i=1 ; i<=$max; i++ )) ; do
	echo "--------------- run = " $i "/" $max "---------------"
	touch "doing_GAN"
	python nn_torch.py
	rm "doing_GAN"
	touch "doing_reaction_energy_calc"
	python calc_reaction_energy.py
	rm "doing_reaction_energy_calc"
	python plot.py
done
touch "doing_finished"

if [ $overwrite_bk = "True" ]; then
	cp surf.json surf_${datanum}_bk.json
	cp reaction_energy.json reaction_energy_${datanum}_bk.json
fi

