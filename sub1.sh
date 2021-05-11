#!/bin/sh
surf_json="surf.json"
num_data=40
max_sub=20
todolist="todolist.txt"
tmpdb="tmp.db"

if [ ! -e $surf_json ]; then
	./sub0.sh $num_data
fi

# read todolist
nline=`cat $todolist | wc -l`

if [ $nline -le $max_sub ]; then
	max=$nline
else
	max=$max_sub
fi

#
# now start calculation
#

# remove trash from previous run
rm stdout* stderr* 2> /dev/null
clean 2> /dev/null
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
	#python calc_reaction_energy.py --id $id
	#python rate.py --id $id
	# or
	qsub run_reaction_energy.sh $id

	sleep 5
done

