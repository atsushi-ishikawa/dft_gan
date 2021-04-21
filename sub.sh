#!/bin/bash

script=run_whisky.sh
maxnum=10
for ((i=0; i<$maxnum; i++)); do
	qsub $script
	sleep 20
done

