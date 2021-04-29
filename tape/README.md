# Pretraining Protein Language Model with Megatron-LM

<!-- TOC -->

- [Pretraining Protein Language Model with Megatron-LM](#pretraining-protein-language-model-with-megatron-lm)
  - [Overview](#overview)
  - [Setup](#setup)
  - [Usage](#usage)
    - [Pretrain and Finetune](#pretrain-and-finetune)
    - [Checkpoint Conversion](#checkpoint-conversion)
  - [Protein Bert Model](#protein-bert-model)
      - [Basic Blocks](#basic-blocks)
      - [Residual Connections](#residual-connections)
  - [Downstream Tasks Performance](#downstream-tasks-performance)
  - [Reference](#reference)

<!-- /TOC -->

## Overview

Our pretraining process, in brief, consists of training a protein language model on unlabeled amino acid chains (`PFAM` dataset), and finetuning on the labeled downstream task datasets. Pretraining code and dataset can be found in [pretrain](../pretrain/) folder. This folder is used for finetuning and performance evaluation.

## Setup

The environment requirements can be found [here](./conda_env).

Base environment is used for performing downstream tasks. 

Torch1.7 environment is optional if you don't use the pretrained model we provide. This is due to the change of serialization strategy between torch1.4 and torch1.7. Therefore, when converting Megatron-ProteinLM's checkpoint to TAPE's checkpoint , you will need a newer torch (version 1.7 or higher) to load Megatron's pretrain model checkpoint.


## Usage

### Pretrain and Finetune

We provide a pretrained model checkpoint for downstream tasks, you can download the checkpoint and finetune on downstream tasks, and you can also pretrain your own model based on your hardware situation.


The finetuning process can be split into two parts.
- Generate a ProteinBertModel and transfer Megatron protein model's parameters to it. We provide an example shell script [megatron-tape-generator.sh](./scripts/shells/megatron-tape-generator.sh). You can use this script like `megatron-tape-generator.sh $MEGA_CKPT` (if you don't set $MEGA_CKPT, it will defaults to $PWD/models/mega).
- Finetune on downstream tasks. `tape_train.py` and `tape_eval.py` are used for finetuning and evaluating the performance of finetuned model. Below is the usage.

```shell
# finetune train
python tape_train.py $BASE_MODEL $DOWNSTREAM_TASK --from_pretrained $PATH_TO_CONVERTED_CHECKPOINT --batch_size $BS --num_train_epochs $NUM_EPOCH --learning_rate $LR --output_dir $PATH_TO_OUTPUT_DIR --warmup_steps $WS

# finetune eval
python tape_eval.py $BASE_MODEL $DOWNSTREAM_TASK $PATH_TO_FINETUNED_MODEL --metrics $METRIC --split $DATA_SPLIT(opt)
# Note that for secondary_structure and remote_homology tasks, you need to set --split parameter, since there are more than one evaluation dataset split.
```

For our pretrain model, $BASE_MODEL should be transformer. Besides, if you find your GPU capacity not sufficient, you can consider setting --gradient_accumulation_steps, which stands for `number of forward passes to make, for each backwards pass`.


There are two more parameters that can make finetune process easier.

1. `--force_save`
- usage `python tape_train.py --force_save $PARAM`.
- This argument is added to shorten the checkpoint generation time (TAPE will save checkpoint each epoch, but it will take hours to finish an epoch in pretraining).
- Default value is 'FALSE' (just save normally, no early saving). If you specify a path as the parameter, a pretrained checkpoint will be saved to there (target checkpoint in checkpoint conversion).

2. `--time_or_name`
- usage `python tape_train.py ... --time_or_name time/$NAME`.
- This argument is used for setting the suffix of checkpoint's name, making it easier to name and find the checkpoints. The naming rule is `task-base_model_name-suffix`. If `time_or_name` is not set (default=time), the suffix will be `timestamp + 6-digit_random_int`; in other cases, the suffix of the checkpoint will be the parameter you passed to `time_or_name`.


### Checkpoint Conversion
Script [`megatron-converter.py`](./converter/megatron-converter.py) is used for transferring the parameters from pretrain language model (Megatron-LM framework) to TAPE.

**Parameter explanation:**

- `src`:      the location of Megatron-ProteinLM's checkpoint.
- `dst`:      the location of TAPE's (random generated) checkpoint.
- `out`:      the location to save the transferred model.
- `dtp`:      default=`torch.float32`(destination data type). Used to specify out model's data type. You can pass parameters like `torch.float16` if you want a fp16 model.
- `hidden`:   hidden size of each encoder layer.
- `heads`:    number of attention heads for each attention layer in the ProteinBert encoder.
- `layers`:   number of hidden layers in the ProteinBert encoder, default=16.

PS: if you meet problems when loading checkpoints (errors related to serialization), one possible solution is `_use_new_zipfile_serialization` of set torch.save() to False.


## Protein Bert Model
We modified layer structure in [modeling_bert.py](./tape/models/modeling_bert.py), dividing it into `four` sub-modules, with `two` residual connections in it.

#### Basic Blocks
1. Input layernorm (LayerNorm)
2. Attention (ProteinBertAttention = ProteinBertSelfAttention + DenseLayer)
3. Post attention layernorm (LayerNorm)
4. FFN layer (ProteinBertFFN, a simple MLP: hidden_size -> 4 * hidden_size -> hidden_size)

#### Residual Connections
- res1: connection1 start - (`INPUT_LAYERNORM`, `ATTENTION`) - connection1 end.
- res2: connection2 start - (`POST_ATTN_LAYERNORM`, `FFN`) - connection2 end.


## Downstream Tasks Performance

| Task | Metric | TAPE Transformer | ProteinLM (ours) |
|:-:|:-:|:-:|:-:|
| contact prediction  | P@L/5               | 0.36 | **0.52** |
| remote_homology     | Top 1 Accuracy      | 0.21 | **0.26** |
| secondary_structure | Accuracy (3-class)  | 0.73 | **0.75** |
| fluorescence        | Spearman's rho      | 0.68 | 0.68 |
| stability           | Spearman's rho      | 0.73 | **0.77** |



## Reference

Our work is based on the following papers.
- [Evaluating Protein Transfer Learning with TAPE](https://arxiv.org/abs/1906.08230v1)
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053v4)

Besides, part of the code is based on [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) and [TAPE](https://github.com/songlab-cal/tape).

__Evaluating Protein Transfer Learning with TAPE__
```
@article{DBLP:journals/corr/abs-1909-08053,
  author    = {Mohammad Shoeybi and
               Mostofa Patwary and
               Raul Puri and
               Patrick LeGresley and
               Jared Casper and
               Bryan Catanzaro},
  title     = {Megatron-LM: Training Multi-Billion Parameter Language Models Using
               Model Parallelism},
  journal   = {CoRR},
  volume    = {abs/1909.08053},
  year      = {2019},
  url       = {http://arxiv.org/abs/1909.08053},
  archivePrefix = {arXiv},
  eprint    = {1909.08053},
  timestamp = {Tue, 24 Sep 2019 11:33:51 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1909-08053.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

__Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism__
```
@article{DBLP:journals/corr/abs-1906-08230,
  author    = {Roshan Rao and
               Nicholas Bhattacharya and
               Neil Thomas and
               Yan Duan and
               Xi Chen and
               John F. Canny and
               Pieter Abbeel and
               Yun S. Song},
  title     = {Evaluating Protein Transfer Learning with {TAPE}},
  journal   = {CoRR},
  volume    = {abs/1906.08230},
  year      = {2019},
  url       = {http://arxiv.org/abs/1906.08230},
  archivePrefix = {arXiv},
  eprint    = {1906.08230},
  timestamp = {Sat, 23 Jan 2021 01:20:25 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1906-08230.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

