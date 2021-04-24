#!/bin/bash
set -xe

# The hyperparameters are copied from 
# https://github.com/songlab-cal/tape/blob/master/config/transformer_config.json

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=protein03
MASTER_PORT=6020
NNODES=$OMPI_COMM_WORLD_SIZE
NODE_RANK=$OMPI_COMM_WORLD_RANK
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
NAME="HIDEEN1024_LAYER16_HEAD16" # NAME should be changed based on the model structure you use.
DATA_PATH=<Specify path form PFAM> # e.g. /protein-data/megatron-data/pfam_all_text_document
CHECKPOINT_PATH=<Specify where to save PFAM-pretrain checkpoints> # e.g. /protein-data/megatron-tape-ckpts/$NAME


mkdir -p $CHECKPOINT_PATH
mkdir -p /protein-data/tape-$NAME-tb # path to save tensorboard's log files

MICRO_BATCH_SIZE=4

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

(python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_tape.py \
       --num-layers 16 \
       --attention-dropout 0.1 \
       --hidden-dropout 0.1 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --layernorm-epsilon 1e-12 \
       --micro-batch-size $MICRO_BATCH_SIZE \
       --global-batch-size $(($MICRO_BATCH_SIZE*$WORLD_SIZE)) \
       --seq-length 2176 \
       --max-position-embeddings 2176 \
       --train-iters 1000000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file <Specify path to iupac_vocab.txt> \
       --data-impl mmap \
       --split 32593668,1715454,44311 \
       --num-workers 96 \
       --distributed-backend nccl \
       --lr 0.0001 \
       --lr-decay-style linear \
       --min-lr 1.0e-5 \
       --lr-decay-iters 990000 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --log-interval 1 \
       --save-interval 1000 \
       --eval-interval 100 \
       --eval-iters 10 \
       --fp16 \
       --checkpoint-activations \
       --tensorboard-dir /protein-data/tape-$NAME-tb
) |& tee -a /protein-data/tape-tb/$NAME.log
