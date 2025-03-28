# MIG-Destroy-Debug
Although NVIDIA states that MIG instances can be created and destroyed dynamically, we notice that, sometimes, even when a MIG instance is idle, it cannot be destroyed. This usually happens with a MIG instance that was previously used and other MIG instances on the same GPU are still in use.

Testing environment:
- NVIDIA-SMI 570.86.15 
- Driver Version: 570.86.15
- CUDA Version: 12.8

 Before the experiment, no process is using GPU (`sudo lsof /dev/nvidia*` shows nothing).

## Reproduce
In the script `./test.sh`, we use `inference_long.py` (long-term job) and `inference_short.py` (short-term job) inside the folder `inference` as examples.

We notice that when `inference_short.py` is finished, the compute instance in MIG 2g can be destroyed, but gpu instance can not.

```
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
```

Output:
```
+ sudo nvidia-smi mig -i 0 -dci -ci 0 -gi 5
Successfully destroyed compute instance ID  0 from GPU  0 GPU instance ID  5
+ sudo nvidia-smi mig -i 0 -dgi -gi 5
Unable to destroy GPU instance ID  5 from GPU  0: In use by another client
Failed to destroy GPU instances: In use by another client
```