#!/bin/sh

### preparation 
#surf_json="surf.json"
num_data=$1
todolist="todolist.txt"
basedir=${HOME}/dft_gan
ciffile="RuO2.cif"

# cleanup
rm $todolist reaction_energy.json loss.h5 2> /dev/null
rm -rf ./log 2> /dev/null

python $basedir/make_surf.py --num_data=$num_data --cif=$basedir/data/$ciffile --elem_from="Ru" --elem_to=["Ir","Pt"] --max_replace_percent=100 --vacuum=7.0
python $basedir/make_todo_list.py
