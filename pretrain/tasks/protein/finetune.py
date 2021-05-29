# coding=utf-8
"""Protein finetuning/evaluation."""

import torch
from megatron import get_args
from megatron import print_rank_0
from megatron import print_rank_last
from megatron import get_tokenizer
from megatron import mpu
from megatron.model.classification import Classification, ClassificationFirstStage, ClassificationIntermediateStage, ClassificationLastStage
from megatron.model.token_classification import TokenClassification, TokenClassificationFirstStage, TokenClassificationIntermediateStage, TokenClassificationLastStage
from tasks.protein.eval_utils import accuracy_func_provider
from tasks.protein.eval_utils import spearmanr_func_provider
from tasks.protein.finetune_utils import protein_classification_forward_step
from tasks.protein.finetune_utils import amino_acid_classification_forward_step
from tasks.protein.finetune_utils import protein_regression_forward_step
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
             forward_step=protein_classification_forward_step,
             end_of_epoch_callback_provider=metrics_func_provider)

def amino_acid_classification(num_classes, Dataset,
                        name_from_datapath_func):
    """Secondary Structure Prediction."""

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
                model = TokenClassificationFirstStage(
                    num_classes=num_classes, num_tokentypes=0)
            elif mpu.is_pipeline_last_stage():
                model = TokenClassificationLastStage(
                    num_classes=num_classes, num_tokentypes=0)
            else:
                model = TokenClassificationIntermediateStage(
                    num_classes=num_classes, num_tokentypes=0)
        else:
            model = TokenClassification(num_classes=num_classes, num_tokentypes=0)

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
             forward_step=amino_acid_classification_forward_step,
             end_of_epoch_callback_provider=metrics_func_provider)


# regression
def protein_regression(num_classes, Dataset,
                        name_from_datapath_func):
    """Stability and Homology Prediction (Regression)."""

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
        # num_classes = 1
        print(f"number of classes = {num_classes}")

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
        return spearmanr_func_provider(single_dataset_provider)

    """Finetune/evaluate."""
    finetune(train_valid_datasets_provider, model_provider,
             forward_step=protein_regression_forward_step,
             end_of_epoch_callback_provider=metrics_func_provider)


def main():
    args = get_args()

    if args.task == 'remote_homology':
        num_classes = 1195
        from tasks.protein.remote_homology import RemoteHomologyDataset as Dataset
        def name_from_datapath(datapath):
            return 'remote_homology'
        protein_classification(num_classes, Dataset, name_from_datapath)
    elif args.task == 'family_accession':
        num_classes = 17929
        # TODO: add family accession
        raise NotImplementedError
    elif args.task == 'secondary_structure':
        num_classes = 3 # {Helix, Strand, Other}
        from tasks.protein.secondary_structure import SecondaryStructureDataset as Dataset
        def name_from_datapath(datapath):
            return 'secondary_structure'
        amino_acid_classification(num_classes, Dataset, name_from_datapath)
    elif args.task == 'stability':
        num_classes = 1 # regression
        from tasks.protein.stability import StabilityDataset as Dataset
        def name_from_datapath(datapath):
            return 'stability'
        protein_regression(num_classes, Dataset, name_from_datapath)
    elif args.task == 'fluorescence':
        num_classes = 1 # regression
        from tasks.protein.fluorescence import FluorescenceDataset as Dataset
        def name_from_datapath(datapath):
            return 'fluorescence'
        protein_regression(num_classes, Dataset, name_from_datapath)
 
    else:
        raise NotImplementedError('Protein task {} is not implemented.'.format(
            args.task))

