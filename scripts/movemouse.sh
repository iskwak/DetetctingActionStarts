#!/bin/bash
xdotool mousemove 160 753
# end is 981
sleep 1
for i in {1..820..1}
do
    echo $i
    x=$(($i + 160))
    xdotool mousemove $x 753
    sleep 0.1
done
