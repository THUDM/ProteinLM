# coding=utf-8
"""Protein finetuning/evaluation."""

import torch
from megatron import get_args
from megatron import print_rank_0
from megatron import print_rank_last
from megatron import get_timers
from megatron import get_tokenizer
from megatron import mpu
from megatron.model.classification import Classification, ClassificationFirstStage, ClassificationIntermediateStage, ClassificationLastStage
from megatron.utils import average_losses_across_data_parallel_group
from tasks.eval_utils import accuracy_func_provider
from tasks.finetune_utils import finetune

def process_batch(batch):
    """Process batch and produce inputs for the model."""
    args = get_args()

    tokens = batch['text'].long().cuda().contiguous()
    labels = batch['label'].long().cuda().contiguous()
    attention_mask = batch['padding_mask'].float().cuda().contiguous()
    max_seq_len = batch['seq_len'].long().max().item()
    max_seq_len = (max_seq_len + 127) // 128 * 128
    if args.fp16:
        attention_mask = attention_mask.half()
    tokens = tokens[:, :max_seq_len]
    attention_mask = attention_mask[:, :max_seq_len]
    return tokens, labels, attention_mask


def protein_classification_forward_step(batch, model, input_tensor):
    """Simple forward step with cross-entropy loss for protein classification."""
    timers = get_timers()

    # Get the batch.
    timers('batch-generator').start()
    try:
        batch_ = next(batch)
    except BaseException:
        batch_ = batch
    tokens, labels, attention_mask = process_batch(batch_)
    timers('batch-generator').stop()

    # Forward model.
    if mpu.is_pipeline_first_stage():
        assert input_tensor is None
        output_tensor = model(tokens, attention_mask, tokentype_ids=None)
    else:
        assert input_tensor is not None
        output_tensor = model(input_tensor, attention_mask)

    if mpu.is_pipeline_last_stage():
        logits = output_tensor

        # Cross-entropy loss.
        loss_func = torch.nn.CrossEntropyLoss()
        loss = loss_func(logits.contiguous().float(), labels)

        # Reduce loss for logging.
        averaged_loss = average_losses_across_data_parallel_group([loss])

        return loss, {'lm loss': averaged_loss[0]}
    return output_tensor



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
