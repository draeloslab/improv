#!/bin/bash

cd $HOME/Desktop/Code/improv/ || exit

source ~/miniforge3/etc/profile.d/conda.sh  
conda activate improv

export PYTHONPATH=$HOME/Desktop/Code/improv:$HOME/Desktop/Code/:$HOME/Desktop/Code/BayesOptim:$PYTHONPATH

