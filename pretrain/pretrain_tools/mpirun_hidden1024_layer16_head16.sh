#!/bin/bash
set -xe

# protein0x: configuration of each machine's `ip address, host name, alias` needs to be added (in /etc/hosts)
# $PATH_PREFIX/pretrain_script.sh: specify path to the pretraining script you want to use.
mpirun --host protein02:1,protein03:1,protein04:1,protein05:1 -np 4 -bind-to none -map-by slot -x NCCL_IB_DISABLE=0 -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_GID_INDEX=3 -x NCCL_NET_GDR_LEVEL=0 -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl_tcp_if_include bond0 -mca btl ^openib bash $PATH_PREFIX/pretrain_script.sh
