#!/bin/sh
### General options
###  specify queue --
#BSUB -q gpuv100
### Mulige gpu clusters gpuv100, gpuk80 og gpuk40
### -- Ask for number of cores
#BSUB -n 4
### -- Specify that the process should be run exclusively on a gpu
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set the job Name --
#BSUB -J finetune 
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 10:00
# request 15GB of system-memory
#BSUB -R "rusage[mem=15GB]"
### User email address
#BSUB -u s183921@student.dtu.dk
### -- send notification at completion--
##BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -oo output_%J.out
#BSUB -eo error_%J.err
# -- end of LSF options --

#Load modules
source env/bin/activate

python3 -m src.models.finetune_CLIP
# python3 -m src.models.CLIP_vectorize