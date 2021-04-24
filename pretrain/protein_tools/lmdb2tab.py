import argparse
import math
from tqdm import tqdm
from Bio.SeqIO.FastaIO import Seq, SeqRecord
from torch.utils.data import Dataset
from typing import Union, List, Tuple, Sequence, Dict, Any, Optional, Collection
from pathlib import Path
import lmdb
import pickle as pkl


"""
This script is based on 
https://github.com/songlab-cal/tape/blob/master/tape/datasets.py
and 
https://github.com/songlab-cal/tape/blob/master/scripts/lmdb_to_fasta.py
"""

parser = argparse.ArgumentParser(description='Convert an lmdb file into a fasta file')
parser.add_argument('--lmdbfile', type=str, help='The lmdb file to convert', default="pfam_valid.lmdb")
parser.add_argument('--tabfile', type=str, help='The fasta file to output', default="pfam_valid.tab")
args = parser.parse_args()

class LMDBDataset(Dataset):
    """Creates a dataset from an lmdb file.
    Args:
        data_file (Union[str, Path]): Path to lmdb file.
        in_memory (bool, optional): Whether to load the full dataset into memory.
            Default: False.
    """

    def __init__(self,
                 data_file: Union[str, Path],
                 in_memory: bool = False):

        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)

        env = lmdb.open(str(data_file), max_readers=1, readonly=True,
                        lock=False, readahead=False, meminit=False)

        with env.begin(write=False) as txn:
            num_examples = pkl.loads(txn.get(b'num_examples'))

        if in_memory:
            cache = [None] * num_examples
            self._cache = cache

        self._env = env
        self._in_memory = in_memory
        self._num_examples = num_examples

    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)

        if self._in_memory and self._cache[index] is not None:
            item = self._cache[index]
        else:
            with self._env.begin(write=False) as txn:
                item = pkl.loads(txn.get(str(index).encode()))
                if 'id' not in item:
                    item['id'] = str(index)
                if self._in_memory:
                    self._cache[index] = item
        return item

dataset = LMDBDataset(args.lmdbfile)

id_fill = math.ceil(math.log10(len(dataset)))

fastafile = args.tabfile
if not fastafile.endswith('.tab'):
    fastafile += '.tab'

with open(fastafile, 'a') as outfile:
    for i, element in enumerate(tqdm(dataset)):
        id_ = element.get('id', str(i).zfill(id_fill))
        if isinstance(id_, bytes):
            id_ = id_.decode()

        primary = element['primary']
        seq = Seq(primary)
        record = SeqRecord(seq, id_)
        outfile.write(record.format('tab'))
