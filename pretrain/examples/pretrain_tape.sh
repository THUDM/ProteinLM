#!/bin/bash

RANK=0
WORLD_SIZE=1
DATA_PATH=<Specify path and file prefix>_text_document
CHECKPOINT_PATH=<Specify path>

rm -rf $CHECKPOINT_PATH

python pretrain_tape.py \
       --num-layers 12 \
       --attention-dropout 0.1 \
       --hidden-dropout 0.1 \
       --hidden-size 768 \
       --num-attention-heads 12 \
       --layernorm-epsilon 1e-12 \
       --micro-batch-size 4 \
       --global-batch-size 8 \
       --seq-length 2176 \
       --max-position-embeddings 2176 \
       --train-iters 2000000 \
       --lr-decay-iters 990000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file <Specify path to iupac_vocab.txt> \
       --tokenizer-type BertWordPieceCase \
       --data-impl mmap \
       --split 32593668,1715454,44311 \
       --lr 0.0001 \
       --min-lr 0.00001 \
       --lr-decay-style linear \
       --lr-warmup-fraction .01 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --log-interval 100 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --fp16
