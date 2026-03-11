#!/bin/bash

cd $HOME/code/improv/ || exit

source ~/miniforge3/etc/profile.d/conda.sh  
conda activate improv

export PYTHONPATH=$HOME/code/improv:$HOME/code/:$HOME/code/BayesOptim:$PYTHONPATH

