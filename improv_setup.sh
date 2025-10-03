#!/bin/bash

cd $HOME/improv/ || exit

source ~/miniforge3/etc/profile.d/conda.sh  
conda activate improv

export PYTHONPATH=$HOME/improv:$HOME/:$HOME/BayesOpt:$PYTHONPATH