#!/bin/bash
# qsub options:
#$ -l gpu=1
#$ -q gpgpu
# First option informs scheduler that the job requires a gpu.
# Second ensures the job is put in the batch job queue, not the interactive queue
 
# Set up the CUDA environment
export CUDA_HOME=/home/s1306752/cuda-install
export CUDNN_HOME=/home/s1306752/cuda
export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH
export CPATH=${CUDNN_HOME}/include:$CPATH
export PATH=${CUDA_HOME}/bin:${PATH}
 
export PYTHON_PATH=$PATH
# Activate the relevant virtual environment:
source resnet-venv/bin/activate
 
#####
# MAKE SURE TO USE THIS, WHATEVER ENVIRONMENT YOU ARE USING.
# The path should point to the location where you saved the gpu_lock_script provided below.
# It's important to include this, as it prevents collisions on the GPUs.
# source ./gpu_lock.sh

export CUDA_VISIBLE_DEVICES='0,1' 
#python -m bin.main train --model mlp_config/models/baseline.py  --config mlp_config/baseline.yml

python resnet/cifar10_download_and_extract.py
python resnet/cifar10_main.py $@ 

# --model_dir
# --learning_rule
# --learning_rate
