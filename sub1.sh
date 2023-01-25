#!/bin/sh
surf_json="surf.json"
num_data=1
max_sub=$num_data
todolist="todolist.txt"
tmpdb="tmp.db"
dash_server="mio"

dir=${HOME}/dft_gan
submit_shell=run_vasp.sh

delete_unfinished=true
use_dash=false
use_queue=true

# ---------------------------------------------------------------
host=`hostname`
if test $host == "whisky" -o $host == "vodka"; then
	echo "NIMS environment"
	stat=qstat
	sub=qsub
elif test $host == "ito-1"; then
	echo "Kyushu university ITO"
	stat=pjstat
	sub=pjsub
elif test $host == "polaire1.hucc"; then
	echo "Hokkaido university GrandChariot"
	stat=pjstat
	sub=pjsub
else
	echo "Unknown environment: stop"
	exit
fi
#
# --- delete unfinished jobs ---
#
if $delete_unfinished && $use_queue; then
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
	max_num=$max_sub
else
	max_num=$nline
fi

#
# --- start calculation ---
#
# remove trash from previous run
#
rm stdout* stderr* 2> /dev/null
rm $submit_shell*.{e,o}[0-9]* 2> /dev/null
rm $tmpdb 2> /dev/null

for ((i=0; i<$max_num; i++)); do
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
	if "$use_queue"; then
		# use queuing system
		echo "$sub $submit_shell $id"

		if [ $host == "ito-1" ] || [ $host == "polaire1.hucc" ] ; then
			echo "OK: $sub $submit_shell -x unique_id=$id"
			$sub $submit_shell -x unique_id=$id
		else
			$sub $submit_shell $id
		fi
		sleep 3
	else
		# do not use queuing system ... direct execution
		python calc_reaction_energy.py --id $id
	fi
done

touch doing_reaction_energy_calc
if "$use_dash"; then
	if [ $host != $dash_server ]; then
		scp doing_reaction_energy_calc $dash_server:$dir
	fi
fi

