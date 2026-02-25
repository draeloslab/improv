#!/bin/bash

lab="$1"
if [ -z "$lab" ]; then
    return 1 2>/dev/null || exit 1
fi


DATE=$(date +%m%d%y)
path="experiments/${lab}"

i=1
while [ -d "${path}/output_${DATE}_ds$i" ]; do
    ((i++))
done

mv ${path}/output "${path}/output_${DATE}_ds$i"
mkdir ${path}/output
echo "Renamed output folder to output_${DATE}_ds$i"

mv ${path}/global.log "${path}/output_${DATE}_ds$i"
echo "Moved global.log to output_${DATE}_ds$i"

# TODO: add bayesopt_param yaml to the folder as well
cp "${path}/bayesopt_parameters.yaml" "${path}/output_${DATE}_ds$i"
echo "Archived parameters to output_${DATE}_ds$i"

if [ -f "${path}/output_${DATE}_ds$i/sample_stream0.h5" ]; then
    cp "${path}/output_${DATE}_ds$i/sample_stream0.h5" ${path}/output/initialization.h5
    echo "Copied sample_stream0 as new initialization file"
elif [ -f "${path}/output_${DATE}_ds$i/sample_stream_end.h5" ]; then
    cp "${path}/output_${DATE}_ds$i/sample_stream_end.h5" ${path}/output/initialization.h5
    echo "Copied sample_stream_end as new initialization file"
else
    echo "Neither sample_stream0 nor sample_stream_end were found in $NEW_FOLDER"
fi
