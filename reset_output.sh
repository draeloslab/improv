#!/bin/bash

DATE=$(date +%m%d%y)

i=1
while [ -d "demos/live/output_${DATE}_ds$i" ]; do
    ((i++))
done

mv demos/live/output "demos/live/output_${DATE}_ds$i"
mkdir demos/live/output
echo "Renamed output folder to output_${DATE}_ds$i"

mv demos/live/global.log "demos/live/output_${DATE}_ds$i"
echo "Moved global.log to output_${DATE}_ds$i"

if [ -f "demos/live/output_${DATE}_ds$i/sample_stream0.h5" ]; then
    cp "demos/live/output_${DATE}_ds$i/sample_stream0.h5" demos/live/output/initialization.h5
    echo "Copied sample_stream0 as new initialization file"
elif [ -f "demos/live/output_${DATE}_ds$i/sample_stream_end.h5" ]; then
    cp "demos/live/output_${DATE}_ds$i/sample_stream_end.h5" demos/live/output/initialization.h5
    echo "Copied sample_stream_end as new initialization file"
else
    echo "Neither sample_stream0 no sample_stream_end were found in output_${DATE}_ds$i"
fi
