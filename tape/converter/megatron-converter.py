#!/usr/bin/env python
# coding: utf-8
from copy import copy
from copy import deepcopy

# # Start Process

import torch
import argparse


parser = argparse.ArgumentParser(description="Transfer Megatron LM's checkpoint to Tape's pretrain model.")
# add args
parser.add_argument('-src', '--src', type=str, help='megatron checkpoint location')
parser.add_argument('-dst', '--dst', type=str, help='tape\'s untrained checkpoint location')
parser.add_argument('-out', '--out', type=str, help='save the transferred model to here')
parser.add_argument('-dtp', '--dtype', type=str, help='destination checkpoint\'s data type, default=fp32. '
                                                      'If you want to specify data type, '
                                                      'use expressions like `torch.float64`')
parser.add_argument('-hidden', '--hidden_dim', type=int, default=1024, help='hidden size of the encoder layers')
parser.add_argument('-heads', '--num_heads', type=int, default=16, help='number of attention heads for each attention '
                                                                        'layer in the ProteinBert encoder')
parser.add_argument('-layers', '--num_layers', type=int, default=16, help='number of hidden layers in the ProteinBert encoder')

args = parser.parse_args()

heads = args.num_heads
hidden_dim = args.hidden_dim
heads_dim = hidden_dim // heads
layers = args.num_layers

# dst's data type
if args.dtype == None:
    DTYPE = torch.float32
else:
    DTYPE = eval(args.dtype)

# load model
lm = torch.load(args.src)['model']['language_model']
emb = lm['embedding']
trans = lm['transformer']
tape = deepcopy(torch.load(args.dst))


def shape_check(dst, src):
    assert dst.shape == src.shape, "shape mismatch"

# Part 1
# ## Embedding Part


# 1.1
# ### Process word-emb
# [PAD] # [MASK] # [CLS] # [SEP] # [UNK] # [unused1] # A
reserved_toks = emb['word_embeddings']['weight'][0:5]
protein_toks = emb['word_embeddings']['weight'][6:31]

toks = torch.cat((reserved_toks, protein_toks))
shape_check(tape['bert.embeddings.word_embeddings.weight'], toks)
tape['bert.embeddings.word_embeddings.weight'] = deepcopy(toks)


# 1.2
# ### Process Pos
shape_check(tape['bert.embeddings.position_embeddings.weight'], emb['position_embeddings']['weight'])
tape['bert.embeddings.position_embeddings.weight'] = emb['position_embeddings']['weight'].clone()

# 1.3
# ### Process token_type
shape_check(tape['bert.embeddings.token_type_embeddings.weight'], torch.zeros_like(tape['bert.embeddings.token_type_embeddings.weight']))
tape['bert.embeddings.token_type_embeddings.weight'] = torch.zeros_like(tape['bert.embeddings.token_type_embeddings.weight']).clone()

# Part 2
# ## Process the bert layers
for layer in range(layers):
    shape_check(tape[f"bert.encoder.layer.{layer}.input_ln.weight"], trans[f"layers.{layer}.input_layernorm.weight"])
    shape_check(tape[f"bert.encoder.layer.{layer}.input_ln.bias"], trans[f"layers.{layer}.input_layernorm.bias"])
    tape[f"bert.encoder.layer.{layer}.input_ln.weight"] = trans[f"layers.{layer}.input_layernorm.weight"].clone()
    tape[f"bert.encoder.layer.{layer}.input_ln.bias"] = trans[f"layers.{layer}.input_layernorm.bias"].clone()

    # attention
    wq, wk, wv = trans[f"layers.{layer}.attention.query_key_value.weight"].clone().view(heads, heads_dim * 3, -1).split(heads_dim, dim=1)
    bq, bk, bv = trans[f"layers.{layer}.attention.query_key_value.bias"].clone().view(heads, heads_dim * 3).split(heads_dim, dim=1)

    shape_check(tape[f"bert.encoder.layer.{layer}.attention.self.query.weight"], wq.contiguous().view(hidden_dim, hidden_dim))
    shape_check(tape[f"bert.encoder.layer.{layer}.attention.self.query.bias"], bq.contiguous().view(-1))

    tape[f"bert.encoder.layer.{layer}.attention.self.query.weight"] = wq.contiguous().view(hidden_dim, hidden_dim).clone()
    tape[f"bert.encoder.layer.{layer}.attention.self.query.bias"] = bq.contiguous().view(-1).clone()
    tape[f"bert.encoder.layer.{layer}.attention.self.key.weight"] = wk.contiguous().view(hidden_dim, hidden_dim).clone()
    tape[f"bert.encoder.layer.{layer}.attention.self.key.bias"] = bk.contiguous().view(-1).clone()
    tape[f"bert.encoder.layer.{layer}.attention.self.value.weight"] = wv.contiguous().view(hidden_dim, hidden_dim).clone()
    tape[f"bert.encoder.layer.{layer}.attention.self.value.bias"] = bv.contiguous().view(-1).clone()

    shape_check(tape[f"bert.encoder.layer.{layer}.attention.dense.weight"], trans[
        f"layers.{layer}.attention.dense.weight"])
    shape_check(tape[f"bert.encoder.layer.{layer}.post_attn_ln.weight"], trans[
        f"layers.{layer}.post_attention_layernorm.weight"])

    tape[f"bert.encoder.layer.{layer}.attention.dense.weight"] = trans[
        f"layers.{layer}.attention.dense.weight"].clone()
    tape[f"bert.encoder.layer.{layer}.attention.dense.bias"] = trans[
        f"layers.{layer}.attention.dense.bias"].clone()
    tape[f"bert.encoder.layer.{layer}.post_attn_ln.weight"] = trans[
        f"layers.{layer}.post_attention_layernorm.weight"].clone()
    tape[f"bert.encoder.layer.{layer}.post_attn_ln.bias"] = trans[
        f"layers.{layer}.post_attention_layernorm.bias"].clone()

    # dense h-4h-h, corresponding to Mega's self.mlp
    shape_check(tape[f"bert.encoder.layer.{layer}.mlp.dense_h_4h.weight"], trans[
        f"layers.{layer}.mlp.dense_h_to_4h.weight"])
    shape_check(tape[f"bert.encoder.layer.{layer}.mlp.dense_4h_h.weight"], trans[f"layers.{layer}.mlp.dense_4h_to_h.weight"])

    tape[f"bert.encoder.layer.{layer}.mlp.dense_h_4h.weight"] = trans[
        f"layers.{layer}.mlp.dense_h_to_4h.weight"].clone()
    tape[f"bert.encoder.layer.{layer}.mlp.dense_h_4h.bias"] = trans[
        f"layers.{layer}.mlp.dense_h_to_4h.bias"].clone()
    tape[f"bert.encoder.layer.{layer}.mlp.dense_4h_h.weight"] = trans[f"layers.{layer}.mlp.dense_4h_to_h.weight"].clone()
    tape[f"bert.encoder.layer.{layer}.mlp.dense_4h_h.bias"] = trans[f"layers.{layer}.mlp.dense_4h_to_h.bias"].clone()


# Part 3
shape_check(tape['bert.encoder.final_ln.weight'], trans['final_layernorm.weight'])
tape['bert.encoder.final_ln.weight'] = trans['final_layernorm.weight'].clone()
tape['bert.encoder.final_ln.bias'] = trans['final_layernorm.bias'].clone()


torch.save(tape, args.out, _use_new_zipfile_serialization=False)
