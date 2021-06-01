import torch
from megatron import get_args
from megatron import get_timers
from megatron import get_tokenizer
from megatron import mpu
from megatron.utils import average_losses_across_data_parallel_group

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
    if labels.dim() == 2:
        # amino acid prediction
        labels = labels[:, :max_seq_len]
    elif labels.dim() == 3:
        # contact prediction
        labels = labels[:, :max_seq_len, :max_seq_len]
    return tokens, labels, attention_mask


def process_batch_regression(batch):
    """Process batch and produce inputs for the model."""
    args = get_args()
    tokens = batch['text'].long().cuda().contiguous()
    labels = batch['label'].cuda().contiguous()
    attention_mask = batch['padding_mask'].float().cuda().contiguous()
    max_seq_len = batch['seq_len'].long().max().item()
    max_seq_len = (max_seq_len + 127) // 128 * 128
    if args.fp16:
        attention_mask = attention_mask.half()
    tokens = tokens[:, :max_seq_len]
    attention_mask = attention_mask[:, :max_seq_len]
    if labels.dim() == 2:
        labels = labels[:, :max_seq_len]
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


def protein_regression_forward_step(batch, model, input_tensor):
    """Simple forward step with MSE loss for protein regression."""
    timers = get_timers()
    # Get the batch.
    timers('batch-generator').start()
    try:
        batch_ = next(batch)
    except BaseException:
        batch_ = batch
    tokens, labels, attention_mask = process_batch(batch_, False)
    timers('batch-generator').stop()
    # Forward model.
    if mpu.is_pipeline_first_stage():
        assert input_tensor is None
        output_tensor = model(tokens, attention_mask, tokentype_ids=None)
    else:
        assert input_tensor is not None
        output_tensor = model(input_tensor, attention_mask)
    if mpu.is_pipeline_last_stage():
        pred_values = output_tensor

        # MSE loss.
        loss_func = torch.nn.MSELoss()
        # print("pred_value = {}, labels = {}. shape = {} {}".format(pred_values.contiguous().float(), labels.float(), pred_values.contiguous().float().shape, labels.float().shape))
        loss = loss_func(pred_values.contiguous().flatten().float(), labels.float())

        # Reduce loss for logging.
        averaged_loss = average_losses_across_data_parallel_group([loss])

        return loss, {'lm loss': averaged_loss[0]}
    return output_tensor



def amino_acid_classification_forward_step(batch, model, input_tensor):
    """Simple forward step with cross-entropy loss for protein classification."""
    timers = get_timers()
    tokenizer = get_tokenizer()

    # Get the batch.
    timers('batch-generator').start()
    try:
        batch_ = next(batch)
    except BaseException:
        batch_ = batch
    tokens, labels, attention_mask = process_batch(batch_)
    assert torch.all(labels[tokens == tokenizer.cls] == -1)
    assert torch.all(labels[tokens == tokenizer.pad] == -1)
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
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1)
        loss = loss_func(logits.contiguous().float(), labels.contiguous().view(-1))

        # Reduce loss for logging.
        averaged_loss = average_losses_across_data_parallel_group([loss])

        return loss, {'lm loss': averaged_loss[0]}
    return output_tensor
