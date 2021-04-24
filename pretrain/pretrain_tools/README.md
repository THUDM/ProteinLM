# Megatron-Protein

This directory contains scripts for [`pretraining Megatron-protein`](./pretrain_tape_distributed_hidden1024_layer16_head16.sh) and [`distributed training script`](./mpirun_hidden1024_layer16_head16.sh).

## Part 1: Pretraining script

You need to modify the path varibles, like `DATA_PATH` (you need to specify the path to your processed_pfam_documents.bin/idx), `iupac_vocab.txt`, etc..

You can also adjust the model's hyperparameters like `hidden size` and `number of attention heads` from the script.


## Part 2: Distributed training

We trained the model on 4 machines, each one with 8 Tesla V100(32G) GPUs.

For parallel training, you need to install mpirun (our version is OpenMPI-4.0.5). You can download Open MPI from [here](https://www.open-mpi.org/software/ompi/v4.0/).

For communication among nodes, you need to set host alias in `/etc/hosts` if your servers are Linux-based.

If you meet timeout problems in distributed training, you can try to set `torch.distributed.init_process_group`'s `timeout` parameter to a longer duration (default value is 30mins) in [`initialize.py`](../megatron/initialize.py).


```python
from datetime import timedelta
# ...
torch.distributed.init_process_group(
    backend=args.distributed_backend,
    world_size=args.world_size, rank=args.rank,
    init_method=init_method, timeout=timedelta(hours=8))
```


## Other Details

### Model parameters
- hidden size = 1024
- number of layers = 16
- number of attention heads = 16

The model checkpoint can be downloaded from [GoogleDrive](https://drive.google.com/file/d/1BkJn_7y7LNWyxntaAPa333jDGIVoTbrs/view?usp=sharing), or [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/f62bef666bc742ebb7c2/?dl=1).

### Time Cost
The pretraining is carried out on 4 machines, each one with 8 Tesla-V100(32G) GPUs for about 5 days. 

Typically, for each iteration, forward compute costs 420 ms, backward compute costs 1000ms.


### Perplexity
The protein model trained in Megatron-LM framework reached a ppl of 5.8.
