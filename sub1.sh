#!/bin/sh
surf_json="surf.json"
numdata=1
todolist="todolist.txt"

if [ ! -e $surf_json ]; then
	# surf_json does not exist ... new calculation

	# cleanup
	rm $todolist reaction_energy.json loss.h5 2> /dev/null
	rm -rf ./log 2> /dev/null

	python make_surf.py $numdata
	python make_todo_list.py
fi

# read todolist
nline=`cat $todolist | wc -l`

# extract id from todlist and delete it
for ((i=0; i<$nline; i++)); do
	id=`head -1 $todolist`
	tail -n +2 $todolist > tmp$$
	mv tmp$$ $todolist
	#
	# do reaction energy calculation for id
	# do rate calculation for id
	#         
	#python calc_reaction_energy.py --id $id
	#python rate.py --id $id
	#
	qsub run_whisky.sh $id

	sleep 5
done

