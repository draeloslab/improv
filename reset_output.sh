#!/bin/bash

DATE=$(date +%m%d%y)
BASE_DIR="experiments/burgess"

i=1
while [ -d "${BASE_DIR}/output_${DATE}_ds$i" ]; do
    ((i++))
done

NEW_FOLDER="${BASE_DIR}/output_${DATE}_ds$i"

mv "${BASE_DIR}/output" "$NEW_FOLDER"
mkdir "${BASE_DIR}/output"
echo "Renamed output folder to output_${DATE}_ds$i"

mv "${BASE_DIR}/global.log" "$NEW_FOLDER"
echo "Moved global.log to output_${DATE}_ds$i"

cp "${BASE_DIR}/bayesopt_parameters.yaml" "$NEW_FOLDER/"
echo "Archived parameters to $NEW_FOLDER"

if [ -f "$NEW_FOLDER/sample_stream0.h5" ]; then
    cp "$NEW_FOLDER/sample_stream0.h5" "${BASE_DIR}/output/initialization.h5"
    echo "Copied sample_stream0 as new initialization file"
elif [ -f "$NEW_FOLDER/sample_stream_end.h5" ]; then
    cp "$NEW_FOLDER/sample_stream_end.h5" "${BASE_DIR}/output/initialization.h5"
    echo "Copied sample_stream_end as new initialization file"
else
    echo "Neither sample_stream0 nor sample_stream_end were found in $NEW_FOLDER"
fi
