#!/bin/bash

todolist="todolist.txt"
num_sample=100

if [ ! -e $todolist ]; then
	python make_surf.py $num_sample
	grep "unique_id" surf.json | cut -d ":" -f 2 | tr -d "\"," > $todolist
fi

id=`head -1 $todolist`
tail -n +2 $todolist > tmp$$
mv tmp$$ $todolist

python calc_reaction_energy.py --id $id
python rate.py --id $id

if [ ! -s $todolist ]; then
	rm $todolist
fi
