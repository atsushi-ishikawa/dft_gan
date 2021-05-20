#!/bin/sh

host=`hostname`
echo $host

if [ $host == "whiskye" ]; then
	sub="qsub"
else
	sub="pjsub"
fi

echo "sub is $sub"
