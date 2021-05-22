"""Remote Homology dataset."""

from megatron import print_rank_0
from .data import ProteinPredictionAbstractDataset
from tasks.data_utils import build_tokens_types_paddings_from_text

class RemoteHomologyDataset(ProteinPredictionAbstractDataset):
    def __init__(self,
                task_name: str,
                dataset_name: str,
                tokenizer,
                max_seq_length: int):
        if split not in ('train', 'valid', 'test_fold_holdout',
                         'test_family_holdout', 'test_superfamily_holdout'):
            raise ValueError(f"Unrecognized split: {split}. Must be one of "
                             f"['train', 'valid', 'test_fold_holdout', "
                             f"'test_family_holdout', 'test_superfamily_holdout']")

        datapaths = [f'remote_homology/remote_homology_{split}.lmdb']
        super().__init__(task_name, dataset_name, datapaths, tokenizer, max_seq_length)


    def build_samples(ids, types, paddings, label, unique_id):
    """Convert to numpy and return a sample consumed by the batch producer."""

        ids_np = np.array(ids, dtype=np.int64)
        types_np = np.array(types, dtype=np.int64)
        paddings_np = np.array(paddings, dtype=np.int64)
        sample = ({'text': ids_np,
                'types': types_np,
                'padding_mask': paddings_np,
                'label': int(label),
                'uid': int(unique_id)})

    return sample
    
    def __getitem__(self, index: int):
        item = self.samples[index]
        ids, types, paddings = build_tokens_types_paddings_from_text(
            item['primary'], None,
            self.tokenizer, self.max_seq_length)
        sample = self.build_samples(ids, types, paddings, item['fold_label'], item['id'])
        return sample