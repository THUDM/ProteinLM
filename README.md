# ProteinLM

We pretrain protein language model based on Megatron-LM framework, and then evaluate the pretrained model results on TAPE (Tasks Assessing Protein Embeddings), which contains a set of five biologically relevant semi-supervised learning tasks. And our pretrained model achieved good performance on these tasks.

# Overview

The proposal of pre-training models such as Bert have greatly promoted the development of natural language processing, improving the performance of language models. Inspired by the similarity of amino acid sequence and text sequence, we consider applying the method of pre-training language model to biological data. 


# Guidance
We provide pretrain and finetune code in two separate folders. If you use the pretrained model we provide, you can simply download the checkpoint and follow the finetune guide. If you want to pretrain your own model yourself, you can refer to the pretrain guide.
- Pretrain [README](./pretrain/README.md)
- Finetune [README](./tape/README.md)


# Project Structure
```
.
├── pretrain                (protein language model pretrain)
│   ├── megatron            (model folder)
│   ├── pretrain_tools      (multi-node pretrain)
│   ├── protein_tools       (data preprocess shells)
└── tape
    ├── conda_env           (conda env in yaml format)
    ├── converter           (converter script and model config files)
    ├── scripts             (model generator, finetune)
    └── tape                (tape model)
```

# Usage

As the structure above shows, there are two stages as follows.

- Pretrain
  - Prepare dataset (`PFAM`)
  - Preprocess data
  - Pretrain
- Finetune
  - Convert pretrain protein model checkpoint
  - Finetune on downstream tasks

Detailed explanations are given in each folder's readme.


# Downstream Tasks Performance

| Task | Metric | TAPE Transformer | ProteinLM (ours) |
|:-:|:-:|:-:|:-:|
| contact prediction  | P@L/5               | 0.36 | **0.52** |
| remote_homology     | Top 1 Accuracy      | 0.21 | **0.26** |
| secondary_structure | Accuracy (3-class)  | 0.73 | **0.75** |
| fluorescence        | Spearman's rho      | 0.68 | 0.68 |
| stability           | Spearman's rho      | 0.73 | **0.77** |


# Reference

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

