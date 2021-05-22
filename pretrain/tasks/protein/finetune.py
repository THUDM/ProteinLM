# coding=utf-8
"""Protein finetuning/evaluation."""

from megatron import get_args
from megatron import print_rank_0
from megatron import get_tokenizer
from megatron import mpu
from megatron.model.classification import Classification, ClassificationFirstStage, ClassificationIntermediateStage, ClassificationLastStage
from tasks.eval_utils import accuracy_func_provider
from tasks.finetune_utils import finetune


def protein_classification(num_classes, Dataset,
                        name_from_datapath_func):
    """Remote Homology Detection and Family Accession Prediction."""

    def train_valid_datasets_provider():
        """Build train and validation dataset."""
        args = get_args()
        tokenizer = get_tokenizer()

        train_dataset = Dataset('training', args.train_data,
                                tokenizer, args.seq_length)
        valid_dataset = Dataset('validation', args.valid_data,
                                tokenizer, args.seq_length)

        return train_dataset, valid_dataset

    def model_provider():
        """Build the model."""
        args = get_args()

        print_rank_0('building classification model for {} ...'.format(
            args.task))
        if mpu.get_pipeline_model_parallel_world_size() > 1:
            # Determine model based on position of stage in pipeline.
            if mpu.is_pipeline_first_stage():
                model = ClassificationFirstStage(
                    num_classes=num_classes, num_tokentypes=0)
            elif mpu.is_pipeline_last_stage():
                model = ClassificationLastStage(
                    num_classes=num_classes, num_tokentypes=0)
            else:
                model = ClassificationIntermediateStage(
                    num_classes=num_classes, num_tokentypes=0)
        else:
            model = Classification(num_classes=num_classes, num_tokentypes=0)

        return model

    def metrics_func_provider():
        """Privde metrics callback function."""
        def single_dataset_provider(datapath):
            args = get_args()
            tokenizer = get_tokenizer()

            name = name_from_datapath_func(datapath)
            return Dataset(name, [datapath], tokenizer, args.seq_length)
        return accuracy_func_provider(single_dataset_provider)

    """Finetune/evaluate."""
    finetune(train_valid_datasets_provider, model_provider,
             end_of_epoch_callback_provider=metrics_func_provider)


def main():
    args = get_args()

    if args.task == 'remote_homology':
        num_classes = 1195
        from tasks.protein.remote_homology import RemoteHomologyDataset as Dataset
        def name_from_datapath(datapath):
            return 'remote_homology'
    elif args.task == 'family_accession':
        num_classes = 17929
        # TODO: add family accession
        raise NotImplementedError
    else:
        raise NotImplementedError('Protein task {} is not implemented.'.format(
            args.task))

    protein_classification(num_classes, Dataset, name_from_datapath)
