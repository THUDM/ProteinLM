from abc import ABC
from abc import abstractmethod

from torch.utils.data import Dataset

from megatron import print_rank_0

def process_samples_from_single_lmdb_path(datapath):
    print_rank_0('   > working on {}'.format(datapath))
    start_time = time.time()
    env = lmdb.open(str(datapath), max_readers=1, readonly=True,
                    lock=False, readahead=False, meminit=False)

    with env.begin(write=False) as txn:
        num_examples = pkl.loads(txn.get(b'num_examples'))

    cache = []
    for index in range(num_examples):
        with env.begin(write=False) as txn:
            item = pkl.loads(txn.get(str(index).encode()))
            if 'id' not in item:
                item['id'] = str(index)
            cache.append(item)
    elapsed_time = time.time() - start_time
    print_rank_0('    > processed {} document, {} questions, and {} samples'
                 ' in {:.2f} seconds'.format(num_docs, num_questions,
                                             num_samples, elapsed_time))
    return cache

class ProteinPredictionAbstractDataset(ABC, Dataset):
    """protein prediction base dataset class, predict the label of a protein seq"""

    def __init__(self, task_name, dataset_name, datapaths,
                 tokenizer, max_seq_length):
        # Store inputs.
        self.task_name = task_name
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        print_rank_0(' > building {} dataset for {}:'.format(self.task_name,
                                                             self.dataset_name))
        # Process the files.
        string = '  > paths:'
        for path in datapaths:
            string += ' ' + path
        print_rank_0(string)
        self.samples = []
        for datapath in datapaths:
            self.samples.extend(process_samples_from_single_lmdb_path(datapath))
        print_rank_0('  >> total number of samples: {}'.format(
            len(self.samples)))

    def __len__(self):
        return len(self.samples)

    @abstractmethod
    def __getitem__(self, idx):
        """Abstract method that returns a single sample"""
        pass
    
    @abstractmethod
    def build_samples(self, idx):
        """Abstract method that returns a single sample"""
        pass

