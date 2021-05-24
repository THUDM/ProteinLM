"""Remote Homology dataset."""

import numpy as np
from megatron import print_rank_0
from .data import ProteinPredictionAbstractDataset
from .data import build_tokens_paddings_from_text
from scipy.spatial.distance import pdist, squareform

class ContactPredictionDataset(ProteinPredictionAbstractDataset):
    def __init__(self,
                name: str,
                datapaths,
                tokenizer,
                max_seq_length: int):
        super().__init__('contact prediction', name, datapaths, tokenizer, max_seq_length)

    def build_samples(self, ids, paddings, tertiary, valid_mask, unique_id, seq_len):
        """Convert to numpy and return a sample consumed by the batch producer."""

        ids_np = np.array(ids, dtype=np.int64)
        paddings_np = np.array(paddings, dtype=np.int64)

        contact_map = np.less(squareform(pdist(tertiary)), 8.0).astype(np.int64)
        yind, xind = np.indices(contact_map.shape)
        invalid_mask = ~(valid_mask[:, None] & valid_mask[None, :])
        invalid_mask |= np.abs(yind - xind) < 6
        contact_map[invalid_mask] = -1

        contact_map = np.pad(contact_map, ((1, 0), (1, 0)), 'constant', constant_values=-1)
        padding_length = self.max_seq_length - contact_map.shape[0]
        if padding_length > 0:
            contact_map = np.pad(contact_map, ((0, padding_length), (0, padding_length)), 'constant', constant_values=-1)
        contact_map = contact_map[:self.max_seq_length, :self.max_seq_length]

        sample = ({'text': ids_np,
                'padding_mask': paddings_np,
                'label': contact_map,
                'uid': int(unique_id), 
                'seq_len': int(seq_len)})

        return sample

    def __getitem__(self, index: int):
        item = self.samples[index]
        ids, paddings, seq_len = build_tokens_paddings_from_text(
            item['primary'], self.tokenizer, self.max_seq_length)
        seq_len = min(seq_len + 1, self.max_seq_length)
        sample = self.build_samples(ids, paddings, item['tertiary'], item['valid_mask'], item['uid'], seq_len)
        return sample

