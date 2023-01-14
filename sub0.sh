#!/bin/sh

### preparation 
#surf_json="surf.json"
numdata=$1
todolist="todolist.txt"

# cleanup
rm $todolist reaction_energy.json loss.h5 2> /dev/null
rm -rf ./log 2> /dev/null

python make_surf.py $numdata
python make_todo_list.py
