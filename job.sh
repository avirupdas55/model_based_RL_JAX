#!/bin/bash --login 
# SGE options (whose lines must begin with #$) 
#$ -cwd # Run the job in the current directory 
# If running on a GPU, add: # #$ -l v100 
# Will give us 1 GPU to use in our job 
# No -pe line hence a serial (1 CPU-core) job # Can instead use 'a100' for the A100 GPUs (if permitted!) 
#$ -l nvidia_a100=1
#$ -pe smp.pe 4
singularity exec --nv docker://avirupdas55/jax:kchua python3 model_based_experiment.py
# qrsh -l a100=1 -pe smp.pe 10 -cwd bash
# singularity run --nv docker://avirupdas55/diffrl:v4
# singularity run --nv docker://avirupdas55/doc:v1 ## for Dichotomy of Control