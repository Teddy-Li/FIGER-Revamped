from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from torch import nn
import torch


class BertForFiger(BertPreTrainedModel):
	def __init__(self, config):
		super().__init__(config)
		self.num_labels = config.num_labels
		self.encode_mode = config.encode_mode
		assert self.encode_mode in ['cls', 'entity', 'left_right', 'cls_entity', 'left_right_entity']

		self.bert = BertModel(config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)

		# TODO: maybe add a linear layer in front of the classifier
		if self.encode_mode in ['cls', 'entity']:
			self.classifier = nn.Linear(config.hidden_size, config.num_labels)
		elif self.encode_mode in ['left_right', 'cls_entity']:
			self.classifier = nn.Linear(2*config.hidden_size, self.num_labels)  # TODO: maybe change to 2 layers
		elif self.encode_mode == 'left_right_entity':
			self.classifier = nn.Linear(3*config.hidden_size, self.num_labels)
		self.init_weights()

	def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
				inputs_embeds=None, left_mask=None, entity_mask=None, right_mask=None, labels=None):
		outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
							position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds)
		sequence_output = outputs[0]
		left_output = sequence_output * left_mask.unsqueeze(-1)
		entity_output = sequence_output * entity_mask.unsqueeze(-1)
		right_output = sequence_output * right_mask.unsqueeze(-1)

		# Do average pooling for each mask
		left_output = left_output.sum(1) / left_mask.sum(1).unsqueeze(-1)
		entity_output = entity_output.sum(1) / entity_mask.sum(1).unsqueeze(-1)
		right_output = right_output.sum(1) / right_mask.sum(1).unsqueeze(-1)
		if self.encode_mode == 'cls':
			output = sequence_output[:, 0, :]
		elif self.encode_mode == 'entity':
			output = entity_output
		elif self.encode_mode == 'cls_entity':
			output = torch.cat([sequence_output[:, 0, :], entity_output], dim=-1)
		elif self.encode_mode == 'left_right':
			output = torch.cat([left_output, right_output], dim=-1)
		elif self.encode_mode == 'left_right_entity':
			output = torch.cat([left_output, entity_output, right_output], dim=-1)
		else:
			raise NotImplementedError
		output = self.dropout(output)

		# TODO: when adding a linear, reflect this change in forward.
		logits = self.classifier(output)
		loss_fct = nn.BCEWithLogitsLoss()
		loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
		outputs = (loss, logits) + outputs[2:]

		return outputs  # (loss), logits, (hidden_states), (attentions)
