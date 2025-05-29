#!/bin/bash

DATE=$(date +%m%d%y)

i=1
while [ -d "demos/live/output_${DATE}_ds$i" ]; do
    ((i++))
done

mv demos/live/output "demos/live/output_${DATE}_ds$i"
mkdir demos/live/output

cp "demos/live/output_${DATE}_ds$i/sample_stream0.h5" demos/live/output/initialization.h5

echo "Renamed output folder to output_${DATE}_ds$i"
echo "Created new output folder with initialization.h5"