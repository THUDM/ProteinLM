# ProteinLM


- [ProteinLM](#proteinlm)
- [Overview](#overview)
- [Guidance](#guidance)
  - [Download ProteinLM](#download-proteinlm)
    - [ProteinLM (200M)](#proteinlm-200m)
    - [ProteinLM (3B)](#proteinlm-3b)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Downstream Tasks Performance](#downstream-tasks-performance)
- [Citation](#citation)
- [Contact](#contact)
- [Reference](#reference)


We pretrain the protein language model based on the Megatron-LM framework and then evaluate the pretrained model results on TAPE (Tasks Assessing Protein Embeddings), which contains a set of five biologically relevant semi-supervised learning tasks. And our pretrained model achieved good performance on these tasks.



# Overview

The proposal of pre-training models such as Bert has greatly promoted the development of natural language processing, improving the performance of language models. Inspired by the similarity of amino acid sequence and text sequence, we consider applying the method of pre-training language model to biological data. 


# Guidance
We provide pretrain and finetune code in two separate folders. Using the pretrained model we provide, you can download the checkpoint and follow the finetune guide. If you want to pretrain your own model yourself, you can refer to the pretrain guide.
- Pretrain [README](./pretrain/README.md)
- Finetune [README](./tape/README.md)

## Download ProteinLM
### ProteinLM (200M) 
For the pretrained model with 200 million parameters,
you can download model checkpoint via [GoogleDrive](https://drive.google.com/file/d/1BkJn_7y7LNWyxntaAPa333jDGIVoTbrs/view?usp=sharing), or [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/f62bef666bc742ebb7c2/?dl=1).

### ProteinLM (3B) 
For the pretrained model with 3 billion parameters,
you can download the model checkpoint from [here](https://resource.wudaoai.cn/).


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
  - Convert pretrained protein model checkpoint
  - Finetune on downstream tasks

Detailed explanations are given in each folder's readme.


# Downstream Tasks Performance

| Task | Metric | TAPE | ProteinLM (200M) | ProteinLM (3B) |  
|:-:|:-:|:-:|:-:|:-:|
| contact prediction  | P@L/5               | 0.36 | 0.52 | **0.75** |
| remote homology     | Top 1 Accuracy      | 0.21 | 0.26 | **0.30** |
| secondary structure | Accuracy (3-class)  | 0.73 | 0.75 | **0.79** |
| fluorescence        | Spearman's rho      | 0.68 | 0.68 | 0.68 |
| stability           | Spearman's rho      | 0.73 | 0.77 | **0.79** |


# Citation
Please cite our paper if you find our work useful for your research. Our paper is can be accessed [here](https://arxiv.org/abs/2108.07435).
```
@misc{xiao2021modeling,
      title={Modeling Protein Using Large-scale Pretrain Language Model}, 
      author={Yijia Xiao and Jiezhong Qiu and Ziang Li and Chang-Yu Hsieh and Jie Tang},
      year={2021},
      eprint={2108.07435},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


# Contact
If you have any problem using ProteinLM, feel free to contact via [mr.yijia.xiao@gmail.com](mailto:mr.yijia.xiao@gmail.com).


# Acknowledgement

Our work is based on the following papers. And part of the code is based on [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) and [TAPE](https://github.com/songlab-cal/tape).
