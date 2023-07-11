# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""


from transformers.models.bert.modeling_bert import *
from scl_loss import SupConLoss

logger = logging.get_logger(__name__)


@add_start_docstrings("""Bert Model with a `language modeling` head on top.""", BERT_START_DOCSTRING)
class BertForMaskedLM(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias", r"cls.predictions.decoder.weight"]

    def __init__(self, config, **kwargs):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )
        self.punctuation_dict = kwargs["punctuation_dict"]
        self.tokenizer = kwargs["tokenizer"]
        self.lambda_value = kwargs["lambda_value"]
        self.num_puncs = len(self.punctuation_dict)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)
        self.cls.predictions.decoder = nn.Linear(config.hidden_size, self.num_puncs, bias=False)
        
        self.punctuation_dict_copy = self.punctuation_dict.copy()
        self.punctuation_dict_copy.pop("O")
        self.mapping = { self.tokenizer.convert_tokens_to_ids(k) :v for k,v in self.punctuation_dict_copy.items() }
        self.mapping[-100] = -100

        # Initialize weights and apply final processing
        self.post_init()

    def mapping_fun(self, x,*y): 
        if x in self.mapping:
            return self.mapping[x]
        else:
            return 0
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        d_model = sequence_output.shape[-1]
        prediction_scores = self.cls(sequence_output)

        # puncs_scores = self.punc_mlm(sequence_output)
        puncs_label = labels.detach().cpu()
        puncs_label = puncs_label.map_(puncs_label, self.mapping_fun).to(labels)
        
        total_loss = None
        masked_lm_loss = None
        if labels is not None:
            ce = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = ce(prediction_scores.view(-1, self.num_puncs), puncs_label.view(-1))

        labels_index = torch.nonzero(puncs_label.view(-1)>=0)
        sequence_output_index = torch.index_select(sequence_output.view(-1,d_model),dim=0,index=labels_index.squeeze())
        sequence_output_index = sequence_output_index.unsqueeze(1)
        labels_index = torch.index_select(puncs_label.view(-1), dim=0, index=labels_index.view(-1))
        scl = SupConLoss()
        scl_loss = scl(sequence_output_index, labels_index)
        total_loss = torch.tensor(1 - self.lambda_value, device=self.device) * masked_lm_loss + torch.tensor(self.lambda_value, device=self.device) * scl_loss
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return MaskedLMOutput(
            loss=total_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
