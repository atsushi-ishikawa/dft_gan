#!/bin/sh
surf_json="surf.json"
num_data=1
max_sub=$num_data
todolist="todolist.txt"
tmpdb="tmp.db"
dash_server="mio"

dir=${HOME}/ase/nn_reac
submit_shell=run_reaction_energy.sh
delete_unfinished=true
use_queue=true

# ---------------------------------------------------------------
host=`hostname`
if [ $host == "whisky" ]; then
	stat=qstat
	sub=qsub
else
	stat=pjstat
	sub=pjsub
fi
#
# --- delete unfinished jobs ---
#
if $delete_unfinished; then
	$stat > tmp$$.txt
	grep $submit_shell tmp$$.txt | awk '{print $1}' | xargs echo
	rm tmp$$.txt 
fi
#
# --- surf_json does not exist... new run ---
#
if [ ! -e $surf_json ]; then
	./sub0.sh $num_data
fi

#
# --- read todolist ---
#
nline=`cat $todolist | wc -l`

if $use_queue && [ $nline -ge $max_sub ]; then
	max=$max_sub
else
	max=$nline
fi

#
# --- start calculation ---
#

# remove trash from previous run
rm stdout* stderr* 2> /dev/null
rm $submit_shell*.{e,o}[0-9]* 2> /dev/null
rm $tmpdb 2> /dev/null

for ((i=0; i<$max; i++)); do
	#
	# extract id from todlist and delete it
	#
	id=`head -1 $todolist`
	tail -n +2 $todolist > tmp$$
	mv tmp$$ $todolist
	#
	# do reaction energy calculation for id
	# do rate calculation for id
	#         
	if $use_queue; then
		$sub $submit_shell $id
		sleep 4
	else
		python calc_reaction_energy.py --id $id
	fi
done

touch doing_reaction_energy_calc
if [ $host != $dash_server ]; then
	scp doing_reaction_energy_calc $dash_server:$dir
fi
