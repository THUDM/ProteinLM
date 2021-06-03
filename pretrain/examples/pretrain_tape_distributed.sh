#!/bin/bash
set -xe

# The hyperparameters are copied from 
# https://github.com/songlab-cal/tape/blob/master/config/transformer_config.json

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6008
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=<Specify path and file prefix>_text_document
CHECKPOINT_PATH=<Specify path>
mkdir -p $CHECKPOINT_PATH

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

(python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_tape.py \
       --num-layers 12 \
       --attention-dropout 0.1 \
       --hidden-dropout 0.1 \
       --hidden-size 768 \
       --num-attention-heads 12 \
       --layernorm-epsilon 1e-12 \
       --micro-batch-size 4 \
       --global-batch-size 32 \
       --seq-length 2176 \
       --max-position-embeddings 2176 \
       --train-iters 1000000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file <Specify path to iupac_vocab.txt> \
       --data-impl mmap \
       --split 32593668,1715454,44311 \
       --distributed-backend nccl \
       --lr 0.0001 \
       --lr-decay-style linear \
       --min-lr 1.0e-5 \
       --lr-decay-iters 990000 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --log-interval 100 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --fp16 \
       --tensorboard-dir <Specify path to tb-log-dir> \
       --checkpoint-activations
) |& tee <pretrain log path>
