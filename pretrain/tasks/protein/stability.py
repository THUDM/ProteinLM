"""Stability dataset."""

import numpy as np
from megatron import print_rank_0
from .data import ProteinPredictionAbstractDataset
from .data import build_tokens_paddings_from_text

class StabilityDataset(ProteinPredictionAbstractDataset):
    def __init__(self,
                name: str,
                datapaths,
                tokenizer,
                max_seq_length: int):
        super().__init__('stability', name, datapaths, tokenizer, max_seq_length)

    def build_samples(self, ids, paddings, label, unique_id, seq_len):
        """Convert to numpy and return a sample consumed by the batch producer."""

        ids_np = np.array(ids, dtype=np.int64)
        paddings_np = np.array(paddings, dtype=np.int64)
        sample = ({'text': ids_np,
                'padding_mask': paddings_np,
                'label': float(label),
                'uid': int(unique_id),
                'seq_len': int(seq_len)})
        return sample

    def __getitem__(self, index: int):
        item = self.samples[index]
        ids, paddings, seq_len = build_tokens_paddings_from_text(
            item['primary'], self.tokenizer, self.max_seq_length)
        seq_len = min(seq_len + 1, self.max_seq_length)
        sample = self.build_samples(ids, paddings, float(item['stability_score'][0]), item['uid'], seq_len)
        return sample

