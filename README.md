# MIG-Destroy-Debug
Although NVIDIA states that MIG instances can be created and destroyed dynamically, we notice that, sometimes, even when a MIG instance is idle, it cannot be destroyed. This usually happens with a MIG instance that was previously used and other MIG instances on the same GPU are still in use.

Testing environment:
- NVIDIA-SMI 570.86.15 
- Driver Version: 570.86.15
- CUDA Version: 12.8

 Before the experiment, no process is using GPU (`sudo lsof /dev/nvidia*` shows nothing).

## Using BlackScholes as examples
> reference: https://github.com/NVIDIA/cuda-samples
- We will use BlackScholes_long (long-term job) and BlackScholes_short (short-term job) as examples.
```
cd cuda-samples/Samples/5_Domain_Specific/BlackScholes
make
```
- Partition MIG instance
```
sudo nvidia-smi mig -i 0 -cgi 19 -C # 1g MIG instance
sudo nvidia-smi mig -i 0 -cgi 9 -C # 3g MIG instance
sudo nvidia-smi mig -lgi
+-------------------------------------------------------+
| GPU instances:                                        |
| GPU   Name             Profile  Instance   Placement  |
|                          ID       ID       Start:Size |
|=======================================================|
|   0  MIG 1g.5gb          19       13          6:1     |
+-------------------------------------------------------+
|   0  MIG 3g.20gb          9        1          0:4     |
+-------------------------------------------------------+
```
- Run BlackScholes_short on 3g.20gb and run BlackScholes_long on 1g.5gb at the same time.
```
CUDA_VISIBLE_DEVICES=<MIG-3g-UUID> ./BlackScholes_short
CUDA_VISIBLE_DEVICES=<MIG-1g-UUID> ./BlackScholes_long
```
- When BlackScholes_short is finished, try to destroy the 3g.20gb MIG slice. This step sometimes succeeds but sometimes does not.
```
$ sudo nvidia-smi mig -i 0 -dci -ci 0 -gi 1 
Successfully destroyed compute instance ID  0 from GPU  0 GPU instance ID  1
$ sudo nvidia-smi mig -i 0 -dgi -gi 1 
Successfully destroyed GPU instance ID  1 from GPU  0
```

```
$ sudo nvidia-smi mig -i 0 -dci -ci 0 -gi 1 
Successfully destroyed compute instance ID  0 from GPU  0 GPU instance ID  1
$ sudo nvidia-smi mig -i 0 -dgi -gi 1 
Unable to destroy GPU instance ID  1 from GPU  0: In use by another client
Failed to destroy GPU instances: In use by another client
```

## Using LLM Inference examples
- We will use inference_long.py (long-term job) and inference_short.py (short-term job) as examples.
```
cd inference
```
- Partition MIG instance
```
sudo nvidia-smi --gpu-reset
sudo nvidia-smi mig -i 0 -cgi 19 -C # 1g MIG instance
sudo nvidia-smi mig -i 0 -cgi 9 -C # 3g MIG instance
sudo nvidia-smi mig -lgi
+-------------------------------------------------------+
| GPU instances:                                        |
| GPU   Name             Profile  Instance   Placement  |
|                          ID       ID       Start:Size |
|=======================================================|
|   0  MIG 1g.5gb          19       13          6:1     |
+-------------------------------------------------------+
|   0  MIG 3g.20gb          9        1          0:4     |
+-------------------------------------------------------+
```
- Run inference_short.py on 3g.20gb and run inference_long.py on 1g.5gb at the same time.
```
docker run -it --cap-add=SYS_ADMIN --gpus device=<MIG-3g-UUID>  --rm -v ./:/workspace irenetht/pytorch-flask-cuda128 python3 inference_short.py

docker run -it --cap-add=SYS_ADMIN --gpus device=<MIG-1g-UUID>  --rm -v ./:/workspace irenetht/pytorch-flask-cuda128 python3 inference_long.py
```
- When inference_short.py is finished, try to destroy the 3g.20gb MIG slice. This step sometimes succeeds but sometimes does not.
```
$ sudo nvidia-smi mig -i 0 -dci -ci 0 -gi 1 
Successfully destroyed compute instance ID  0 from GPU  0 GPU instance ID  1
$ sudo nvidia-smi mig -i 0 -dgi -gi 1 
Successfully destroyed GPU instance ID  1 from GPU  0
```

```
$ sudo nvidia-smi mig -i 0 -dci -ci 0 -gi 1 
Successfully destroyed compute instance ID  0 from GPU  0 GPU instance ID  1
$ sudo nvidia-smi mig -i 0 -dgi -gi 1
Unable to destroy GPU instance ID  1 from GPU  0: In use by another client
Failed to destroy GPU instances: In use by another client
```