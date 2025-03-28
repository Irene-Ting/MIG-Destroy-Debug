#!/bin/bash
set -x 

cd inference

# Reset GPU
sudo nvidia-smi --gpu-reset

# Create MIG 2G instance
sudo nvidia-smi mig -i 0 -cgi 14 -C 

# Run inference_short.py in the background
export MIG_2G=$(nvidia-smi -L | grep 'MIG 2g.10gb' | awk -F 'UUID: ' '{print $2}' | tr -d ')')
docker run -d --rm --cap-add=SYS_ADMIN --gpus device=$MIG_2G -v ./:/workspace --name short_container irenetht/pytorch-flask-cuda128 python3 inference_short.py

# Create MIG 1G instance
sudo nvidia-smi mig -i 0 -cgi 19 -C 
export MIG_1G=$(nvidia-smi -L | grep 'MIG 1g.5gb' | awk -F 'UUID: ' '{print $2}' | tr -d ')')

# Run inference_long.py in the background
docker run -d --rm --cap-add=SYS_ADMIN --gpus device=$MIG_1G -v ./:/workspace --name long_container irenetht/pytorch-flask-cuda128 python3 inference_long.py

# Wait for the short container to finish
docker wait short_container 

# Destroy MIG 2g instance
sudo nvidia-smi mig -i 0 -dci -ci 0 -gi 5
sudo nvidia-smi mig -i 0 -dgi -gi 5