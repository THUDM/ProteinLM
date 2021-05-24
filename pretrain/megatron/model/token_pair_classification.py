# coding=utf-8

"""Token Classification model."""

import torch

from megatron import get_args, print_rank_last
from megatron import mpu
from megatron.model.bert_model import bert_attention_mask_func, bert_extended_attention_mask, bert_position_ids
from megatron.model.language_model import get_language_model
from megatron.model.utils import get_linear_layer
from megatron.model.utils import init_method_normal
from megatron.model.utils import scaled_init_method_normal
from .module import MegatronModule


class PairwisePredictionHead(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        args = get_args()
        init_method = init_method_normal(args.init_method_std)
        self.classifier = torch.nn.Sequential(
            get_linear_layer(args.hidden_size * 2, args.hidden_size, init_method),
            torch.nn.Tanh(),
            torch.nn.Dropout(args.hidden_dropout),
            get_linear_layer(args.hidden_size, num_classes, init_method))
    def forward(self, lm_output):
        seq_len = lm_output.size(1)
        lm_output_a = lm_output.unsqueeze(1).expand(-1, seq_len, -1, -1) # b x s x s x h
        lm_output_b = lm_output.unsqueeze(2).expand(-1, -1, seq_len, -1) # b x s x s x h
        pair_emb = torch.cat([lm_output_a, lm_output_b], dim=-1) # b x s x s x (2h)
        logits = self.classifier(pair_emb)
        logits = (logits + logits.transpose(1,2)) / 2.0
        return logits


class TokenPairClassificationBase(MegatronModule):

    def __init__(self, num_classes, num_tokentypes=2):
        super(TokenPairClassificationBase, self).__init__(share_word_embeddings=False)
        args = get_args()

        self.num_classes = num_classes
        init_method = init_method_normal(args.init_method_std)

        self.language_model, self._language_model_key = get_language_model(
            attention_mask_func=bert_attention_mask_func,
            num_tokentypes=num_tokentypes,
            add_pooler=False,
            init_method=init_method,
            scaled_init_method=scaled_init_method_normal(args.init_method_std,
                                                         args.num_layers))

        # Multi-choice head.
        if mpu.is_pipeline_last_stage():
            self.classification_head = PairwisePredictionHead(num_classes)
            self._classification_head_key = 'classification_head'

    def forward(self, model_input, attention_mask, tokentype_ids=None):

        extended_attention_mask = bert_extended_attention_mask(attention_mask)

        kwargs = {}
        if mpu.is_pipeline_first_stage():
            input_ids = model_input
            position_ids = bert_position_ids(input_ids)

            args = [input_ids, position_ids, extended_attention_mask]
            kwargs['tokentype_ids'] = tokentype_ids
        else:
            args = [model_input, extended_attention_mask]
        lm_output = self.language_model(*args, **kwargs)
        if mpu.is_pipeline_last_stage():
            classification_logits = self.classification_head(lm_output)

            # Reshape back to separate choices.
            # classification_logits = classification_logits.view(-1, self.num_classes)

            return classification_logits
        return lm_output

    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):
        """For easy load when model is combined with other heads,
        add an extra key."""

        state_dict_ = {}
        state_dict_[self._language_model_key] \
            = self.language_model.state_dict_for_save_checkpoint(
                destination, prefix, keep_vars)
        if mpu.is_pipeline_last_stage():
            state_dict_[self._classification_head_key] \
                = self.classification_head.state_dict(
                    destination, prefix, keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        self.language_model.load_state_dict(
            state_dict[self._language_model_key], strict=strict)
        if mpu.is_pipeline_last_stage():
            if self._classification_head_key in state_dict:
                self.classification_head.load_state_dict(
                    state_dict[self._classification_head_key], strict=strict)
            else:
                print_rank_last('***WARNING*** could not find {} in the checkpoint, '
                                'initializing to random'.format(
                                    self._classification_head_key))


class TokenPairClassification(TokenPairClassificationBase):

    def __init__(self, num_classes, num_tokentypes=2):
        super(TokenPairClassification, self).__init__(
            num_classes, num_tokentypes=num_tokentypes)

    def forward(self, input_ids, attention_mask,
                tokentype_ids=None):
        return super(TokenPairClassification, self).forward(
            input_ids,
            attention_mask,
            tokentype_ids=tokentype_ids)


class TokenPairClassificationFirstStage(TokenPairClassificationBase):

    def __init__(self, num_classes, num_tokentypes=2):
        super(TokenPairlassificationFirstStage, self).__init__(
            num_classes, num_tokentypes=num_tokentypes)

    def forward(self, input_ids, attention_mask,
                tokentype_ids=None):
        return super(TokenPairClassificationFirstStage, self).forward(
            input_ids,
            attention_mask,
            tokentype_ids=tokentype_ids)


class TokenPairClassificationIntermediateStage(TokenPairClassificationBase):

    def __init__(self, num_classes, num_tokentypes=2):
        super(TokenPairClassificationIntermediateStage, self).__init__(
            num_classes, num_tokentypes=num_tokentypes)

    def forward(self, hidden_state, attention_mask):
        return super(TokenPairClassificationIntermediateStage, self).forward(
            hidden_state,
            attention_mask)


class TokenPairClassificationLastStage(TokenPairClassificationBase):

    def __init__(self, num_classes, num_tokentypes=2):
        super(TokenPairClassificationLastStage, self).__init__(
            num_classes, num_tokentypes=num_tokentypes)

    def forward(self, hidden_state, attention_mask):
        return super(TokenPairClassificationLastStage, self).forward(
            hidden_state,
            attention_mask)
