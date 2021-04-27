#!/bin/bash

cwd=`pwd`
script=tmp.sh
maxnum=100

for ((i=0; i<$maxnum; i++)); do
	#qsub $script
	$cwd/$script
	sleep 2
done

