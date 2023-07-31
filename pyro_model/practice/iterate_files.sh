#!/bin/bash

declare -a files=("f1" "f2" "f3")

for f in "${files[@]}"
do
	echo $f
done