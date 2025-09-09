#!/bin/bash

DATE=$(date +%m%d%y)

i=1
while [ -d "experiments/savier/output_${DATE}_ds$i" ]; do
    ((i++))
done

mv experiments/savier/output "experiments/savier/output_${DATE}_ds$i"
mkdir experiments/savier/output
echo "Renamed output folder to output_${DATE}_ds$i"

mv experiments/savier/global.log "experiments/savier/output_${DATE}_ds$i"
echo "Moved global.log to output_${DATE}_ds$i"

if [ -f "experiments/savier/output_${DATE}_ds$i/sample_stream0.h5" ]; then
    cp "experiments/savier/output_${DATE}_ds$i/sample_stream0.h5" experiments/savier/output/initialization.h5
    echo "Copied sample_stream0 as new initialization file"
elif [ -f "experiments/savier/output_${DATE}_ds$i/sample_stream_end.h5" ]; then
    cp "experiments/savier/output_${DATE}_ds$i/sample_stream_end.h5" experiments/savier/output/initialization.h5
    echo "Copied sample_stream_end as new initialization file"
else
    echo "Neither sample_stream0 no sample_stream_end were found in output_${DATE}_ds$i"
fi
