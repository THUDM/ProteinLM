"""Protein Evaluation utilities."""

import os
import time
import scipy.stats

import torch

from megatron import get_args
from megatron import get_tokenizer
from megatron import print_rank_last, is_last_rank
from megatron import mpu
from megatron.training import communicate
from tasks.finetune_utils import build_data_loader
from tasks.protein.finetune_utils import process_batch
from tasks.protein.finetune_utils import compute_precision_at_l5
from tasks.protein.finetune_utils import process_batch_regression


def accuracy_func_provider(single_dataset_provider):
    """Provide function that calculates accuracies."""
    args = get_args()

    # Build dataloaders.
    datapaths = args.valid_data
    dataloaders = []
    for datapath in datapaths:
        dataset = single_dataset_provider(datapath)
        dataloader = build_data_loader(
            dataset, args.micro_batch_size, num_workers=args.num_workers,
            drop_last=(mpu.get_data_parallel_world_size() > 1))
        dataloaders.append((dataset.dataset_name, dataloader))

    def metrics_func(model, epoch, output_predictions=False):
        print_rank_last('calculating metrics ...')
        correct = 0
        total = 0
        if output_predictions:
            assert mpu.get_data_parallel_world_size() == 1
            named_predictions = []
            names = 'predictions'
        for name, dataloader in dataloaders:
            output = calculate_correct_answers(name, model, dataloader,
                                               epoch, output_predictions)
            if not output_predictions:
                correct_ans, total_count = output
            else:
                correct_ans, total_count, predictions = output
                named_predictions.append((name, predictions))
                names += '_' + name
            correct += correct_ans
            total += total_count
        if is_last_rank():
            percent = float(correct) * 100.0 / float(total)
            print(' >> |epoch: {}| overall: correct / total = {} / {} = '
                  '{:.4f} %'.format(epoch, correct, total, percent))

        if output_predictions and is_last_rank():
            assert args.load is not None
            filename = os.path.join(args.load, names + '.pt')
            torch.save(named_predictions, filename)

    return metrics_func


def spearmanr_func_provider(single_dataset_provider):
    """Provide function that calculates spearmanr."""
    args = get_args()

    # Build dataloaders.
    datapaths = args.valid_data
    dataloaders = []
    for datapath in datapaths:
        dataset = single_dataset_provider(datapath)
        dataloader = build_data_loader(
            dataset, args.micro_batch_size, num_workers=args.num_workers,
            drop_last=(mpu.get_data_parallel_world_size() > 1))
        dataloaders.append((dataset.dataset_name, dataloader))

    def metrics_func(model, epoch, output_predictions=False):
        print_rank_last('calculating metrics ...')
        fact_all = list()
        pred_all = list()

        if output_predictions:
            assert mpu.get_data_parallel_world_size() == 1
            named_predictions = []
            names = 'predictions'
        for name, dataloader in dataloaders:
            output = calculate_correct_spearsonr(name, model, dataloader,
                                               epoch, output_predictions)

            if not output_predictions:
                spearmanr_score, pred_values, fact_values = output
            else:
                spearmanr_score, pred_values, fact_values = output
            pred_all.extend(pred_values)
            fact_all.extend(fact_values)

        if is_last_rank():
            pred_all = [i.item() for i in pred_all]
            fact_all = [i.item() for i in fact_all]
            spearmanr_score = scipy.stats.spearmanr(pred_all, fact_all).correlation
            print(' >> |epoch: {}| overall: spearmanr = '
                  '{:.4f} '.format(epoch, spearmanr_score))

        if output_predictions and is_last_rank():
            assert args.load is not None
            filename = os.path.join(args.load, names + '.pt')
            torch.save(named_predictions, filename)

    return metrics_func


def calculate_correct_spearsonr(name, model, dataloader,
                              epoch, output_predictions):
    """Calculate Spearson rank correlation for regression tasks"""
    args = get_args()
    tokenizer = get_tokenizer()
    start_time = time.time()
    model.eval()
    saved_batch_size = args.micro_batch_size
    with torch.no_grad():
        # For all the batches in the dataset.
        # store predicted values and ground-truth values (regression tasks)
        pred_values = []
        fact_values = []
        if output_predictions:
            # This option is only possible when data parallel size is 1.
            assert mpu.get_data_parallel_world_size() == 1
            output_ids = []
            output_preds = []
            output_facts = []

        for _, batch in enumerate(dataloader):
            # Run the model forward.
            tokens, labels_, attention_mask = process_batch_regression(batch)
            if labels_.dim() == 2:
                assert torch.all(labels_[tokens == tokenizer.cls] == -1)
                assert torch.all(labels_[tokens == tokenizer.pad] == -1)

            # For evaluation only mode we use drop_last = False to get all the
            # samples, which means we might not have a full batch, so we
            # adjust batch_size here to actual batch size of data
            actual_batch_size = len(labels_)
            # ... applying sample_multiplier if necessary
            ds = dataloader.dataset
            if hasattr(ds, 'sample_multiplier'):
                actual_batch_size *= ds.sample_multiplier
            args.micro_batch_size = actual_batch_size

            if not mpu.is_pipeline_first_stage():
                input_tensor, _ = communicate(
                    tensor_send_next=None,
                    tensor_send_prev=None,
                    recv_forward=True,
                    recv_backward=False)
            else:
                input_tensor = None

            # Forward model.
            if mpu.is_pipeline_first_stage():
                assert input_tensor is None
                output_tensor = model(tokens, attention_mask, tokentype_ids=None)
            else:
                assert input_tensor is not None
                output_tensor = model(input_tensor, attention_mask)

            if mpu.is_pipeline_last_stage():
                reg_preds = output_tensor
                reg_facts = labels_
                # Add output predictions.
                pred_values.extend(reg_preds.view(-1))
                fact_values.extend(reg_facts.view(-1))
            else:
                communicate(
                    tensor_send_next=output_tensor,
                    tensor_send_prev=None,
                    recv_forward=False,
                    recv_backward=False)

    model.train()
    args.micro_batch_size = saved_batch_size

    num_samples = len(fact_values)
    # Reduce.
    if mpu.is_pipeline_last_stage():
        gathered_preds = [i.item() for i in pred_values]
        gathered_facts = [i.item() for i in fact_values]
        spearmanr_score = scipy.stats.spearmanr(gathered_preds, gathered_facts).correlation
        elapsed_time = time.time() - start_time
        print_rank_last(' > |epoch: {}| name = {}| spearsonr metrics = {:.4f} , elapsed time (sec): {:.3f}'.format(
                            epoch, name, spearmanr_score, elapsed_time))

        if output_predictions:
            return spearmanr_score, pred_values, fact_values
        return spearmanr_score, pred_values, fact_values
    if output_predictions:
        return 0, torch.tensor([]), torch.tensor([])
    return 0




def calculate_correct_pearson(name, model, dataloader,
                              epoch, output_predictions):
    """Calculate pearson correlation for regression tasks"""
    args = get_args()
    tokenizer = get_tokenizer()
    start_time = time.time()
    model.eval()
    saved_batch_size = args.micro_batch_size
    with torch.no_grad():
        # For all the batches in the dataset.
        sigma_x, sigma_x2, sigma_y, sigma_y2, sigma_xy = 0, 0, 0, 0, 0
        pred_values = []
        fact_values = []
        if output_predictions:
            # This option is only possible when data parallel size is 1.
            assert mpu.get_data_parallel_world_size() == 1
        for _, batch in enumerate(dataloader):
            # Run the model forward.
            tokens, labels_, attention_mask = process_batch_regression(batch)
            if labels_.dim() == 2:
                assert torch.all(labels_[tokens == tokenizer.cls] == -1)
                assert torch.all(labels_[tokens == tokenizer.pad] == -1)

            # For evaluation only mode we use drop_last = False to get all the
            # samples, which means we might not have a full batch, so we
            # adjust batch_size here to actual batch size of data
            actual_batch_size = len(labels_)
            # ... applying sample_multiplier if necessary
            ds = dataloader.dataset
            if hasattr(ds, 'sample_multiplier'):
                actual_batch_size *= ds.sample_multiplier
            args.micro_batch_size = actual_batch_size

            if not mpu.is_pipeline_first_stage():
                input_tensor, _ = communicate(
                    tensor_send_next=None,
                    tensor_send_prev=None,
                    recv_forward=True,
                    recv_backward=False)
            else:
                input_tensor = None

            # Forward model.
            if mpu.is_pipeline_first_stage():
                assert input_tensor is None
                output_tensor = model(tokens, attention_mask, tokentype_ids=None)
            else:
                assert input_tensor is not None
                output_tensor = model(input_tensor, attention_mask)

            if mpu.is_pipeline_last_stage():
                reg_preds = output_tensor
                reg_facts = labels_
                # Add output predictions.
                if output_predictions:
                    pred_values.extend(reg_preds.view(-1))
                    fact_values.extend(reg_facts.view(-1))
                sigma_x += sum(reg_preds)
                sigma_y += sum(reg_facts)
                sigma_x2 += sum([i.item() ** 2 for i in reg_preds])
                sigma_y2 += sum([i.item() ** 2 for i in reg_facts])
                sigma_xy += sum([i.item() * j.item() for i, j in zip(reg_preds, reg_facts)])
            else:
                communicate(
                    tensor_send_next=output_tensor,
                    tensor_send_prev=None,
                    recv_forward=False,
                    recv_backward=False)

    model.train()
    args.micro_batch_size = saved_batch_size

    num_samples = len(fact_values)
    # Reduce.
    if mpu.is_pipeline_last_stage():
        unreduced = torch.cuda.FloatTensor([sigma_x, sigma_y, sigma_x2, sigma_y2, sigma_xy, num_samples])

        torch.distributed.all_reduce(unreduced,
                                     group=mpu.get_data_parallel_group())

        # Print on screen.
        tot_x, tot_y, tot_x2, tot_y2, tot_xy, num_samples = unreduced[0], unreduced[1], unreduced[2], unreduced[3], unreduced[4], unreduced[5]
        pearman_r = (num_samples * tot_xy - tot_x * tot_y) / ((num_samples * tot_x2 - (tot_x ** 2)) ** 0.5 * (num_samples * tot_y2 - (tot_y ** 2)) ** 0.5)
        elapsed_time = time.time() - start_time
        print_rank_last(' > |epoch: {}| metrics for {}: tot_preds / tot_facts '
                        '= {} / {} = {:.4f} %, elapsed time (sec): {:.3f}'.format(
                            epoch, name, tot_x, tot_y, 
                            pearman_r, elapsed_time))

        if output_predictions:
            return pearman_r, pred_values, fact_values
        return pearman_r, pred_values, fact_values
    if output_predictions:
        return 0, torch.tensor([]), torch.tensor([])
    return 0


def calculate_correct_answers(name, model, dataloader,
                              epoch, output_predictions):
    """Calculate correct over total answers and return prediction if the
    `output_predictions` is true."""
    args = get_args()
    tokenizer = get_tokenizer()
    start_time = time.time()
    model.eval()
    saved_batch_size = args.micro_batch_size
    with torch.no_grad():
        # For all the batches in the dataset.
        total = 0
        correct = 0
        if output_predictions:
            # This option is only possible when data parallel size is 1.
            assert mpu.get_data_parallel_world_size() == 1
            softmaxes = []
            labels = []
            ids = []
        for _, batch in enumerate(dataloader):
            # Run the model forward.
            tokens, labels_, attention_mask = process_batch(batch)
            seq_len = batch['seq_len']
            if labels_.dim() == 2:
                assert torch.all(labels_[tokens == tokenizer.cls] == -1)
                assert torch.all(labels_[tokens == tokenizer.pad] == -1)

            # For evaluation only mode we use drop_last = False to get all the
            # samples, which means we might not have a full batch, so we
            # adjust batch_size here to actual batch size of data
            actual_batch_size = len(labels_)
            # ... applying sample_multiplier if necessary
            ds = dataloader.dataset
            if hasattr(ds, 'sample_multiplier'):
                actual_batch_size *= ds.sample_multiplier
            args.micro_batch_size = actual_batch_size

            if not mpu.is_pipeline_first_stage():
                input_tensor, _ = communicate(
                    tensor_send_next=None,
                    tensor_send_prev=None,
                    recv_forward=True,
                    recv_backward=False)
            else:
                input_tensor = None

            # Forward model.
            if mpu.is_pipeline_first_stage():
                assert input_tensor is None
                output_tensor = model(tokens, attention_mask, tokentype_ids=None)
            else:
                assert input_tensor is not None
                output_tensor = model(input_tensor, attention_mask)

            if mpu.is_pipeline_last_stage():
                logits = output_tensor

                # Add output predictions.
                if output_predictions:
                    softmaxes.extend(torch.nn.Softmax(dim=-1)(
                        logits.float()).data.cpu().numpy().tolist())
                    labels.extend(labels_.data.cpu().numpy().tolist())
                    ids.extend(batch['uid'].cpu().numpy().tolist())

                if labels_.dim() == 3:
                    # contact prediction
                    res = compute_precision_at_l5(seq_len,
                            logits.contiguous(),
                            labels_.contiguous(),
                            return_precision=False)
                    correct += res[0]
                    total += res[1]
                else:
                    # Compute the correct answers.
                    predicted = torch.argmax(logits, dim=-1).view(-1)
                    labels_flat = labels_.contiguous().view(-1)
                    predicted_actual = predicted[labels_flat != -1]
                    labels_actual = labels_flat[labels_flat != -1]
                    #corrects = (predicted == labels_)
                    corrects = (predicted_actual == labels_actual)
                    # Add to the counters.
                    #total += labels_.size(0)
                    total += labels_actual.size(0)
                    correct += corrects.sum().item()
            else:
                communicate(
                    tensor_send_next=output_tensor,
                    tensor_send_prev=None,
                    recv_forward=False,
                    recv_backward=False)

    model.train()
    args.micro_batch_size = saved_batch_size

    # Reduce.
    if mpu.is_pipeline_last_stage():
        unreduced = torch.cuda.LongTensor([correct, total])
        torch.distributed.all_reduce(unreduced,
                                     group=mpu.get_data_parallel_group())

        # Print on screen.

        correct_ans = unreduced[0].item()
        total_count = unreduced[1].item()
        percent = float(correct_ans) * 100.0 / float(total_count)
        elapsed_time = time.time() - start_time
        print_rank_last(' > |epoch: {}| metrics for {}: correct / total '
                        '= {} / {} = {:.4f} %, elapsed time (sec): {:.3f}'.format(
                            epoch, name, correct_ans, total_count,
                            percent, elapsed_time))

        if output_predictions:
            return correct_ans, total_count, (softmaxes, labels, ids)
        return correct_ans, total_count
    if output_predictions:
        return 0, 0, ()
    return 0, 0

