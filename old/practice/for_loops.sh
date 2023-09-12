#!/bin/bash
k_max=5

for k in $(eval echo "{1..$k_max}")
do
    for r in $(eval echo "{1..$k}")
    do
    	echo "k: $k, r: $r"
    done
done

# for k in range(1, k_max)
# for r in range(1, k)
	# print(k=blah, r=blah)