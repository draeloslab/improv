#!/bin/bash

DATE=$(date +%m%d%y)

i=1
while [ -d "experiments/burgess/output_${DATE}_ds$i" ]; do
    ((i++))
done

mv experiments/burgess/output "experiments/burgess/output_${DATE}_ds$i"
mkdir experiments/burgess/output
echo "Renamed output folder to output_${DATE}_ds$i"

mv experiments/burgess/global.log "experiments/burgess/output_${DATE}_ds$i"
echo "Moved global.log to output_${DATE}_ds$i"

if [ -f "experiments/burgess/output_${DATE}_ds$i/sample_stream0.h5" ]; then
    cp "experiments/burgess/output_${DATE}_ds$i/sample_stream0.h5" experiments/burgess/output/initialization.h5
    echo "Copied sample_stream0 as new initialization file"
elif [ -f "experiments/burgess/output_${DATE}_ds$i/sample_stream_end.h5" ]; then
    cp "experiments/burgess/output_${DATE}_ds$i/sample_stream_end.h5" experiments/burgess/output/initialization.h5
    echo "Copied sample_stream_end as new initialization file"
else
    echo "Neither sample_stream0 no sample_stream_end were found in output_${DATE}_ds$i"
fi
