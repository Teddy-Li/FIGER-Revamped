from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from torch import nn
from torch.nn import functional as F
from torchvision.ops import MLP
import torch


class BertForFiger(BertPreTrainedModel):
	def __init__(self, config):
		super().__init__(config)
		self.num_labels = config.num_labels
		self.encode_mode = config.encode_mode
		assert self.encode_mode in ['cls', 'entity', 'left_right', 'cls_entity', 'left_right_entity']

		self.bert = BertModel(config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
		print(f"encode_mode: {self.encode_mode}; number of classification layers: {config.num_clsf_layers}")

		self.hdszs = [config.hidden_size for x in range(config.num_clsf_layers-1)] + [config.num_labels]

		# TODO: maybe add a linear layer in front of the classifier
		if self.encode_mode in ['cls', 'entity']:
			if config.num_clsf_layers == 1:
				self.classifier = nn.Linear(config.hidden_size, config.num_labels)
			else:
				self.classifier = MLP(config.hidden_size, self.hdszs, norm_layer=torch.nn.LayerNorm,
								  activation_layer=torch.nn.GELU, dropout=0.1)
		elif self.encode_mode in ['left_right', 'cls_entity']:
			if config.num_clsf_layers == 1:
				self.classifier = nn.Linear(config.hidden_size, config.num_labels)
			else:
				self.classifier = MLP(2*config.hidden_size, self.hdszs, norm_layer=torch.nn.LayerNorm,
								  activation_layer=torch.nn.GELU, dropout=0.1)
		elif self.encode_mode == 'left_right_entity':
			if config.num_clsf_layers == 1:
				self.classifier = nn.Linear(config.hidden_size, config.num_labels)
			else:
				self.classifier = MLP(3*config.hidden_size, self.hdszs, norm_layer=torch.nn.LayerNorm,
								  activation_layer=torch.nn.GELU, dropout=0.1)
		self.init_weights()

	def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
				inputs_embeds=None, left_mask=None, entity_mask=None, right_mask=None, labels=None, sentid=None,
				id=None, fileid=None, entity_name=None):
		# TODO: for some reason the F-1 score is always 0.0 when the encode_mode is 'left_right' or 'left_right_entity'.
		# TODO: eval loss there is NAN. Maybe the problem is in the loss function or the logit output.

		outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
							position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds)
		sequence_output = outputs[0]
		left_output = sequence_output * left_mask.unsqueeze(-1)
		entity_output = sequence_output * entity_mask.unsqueeze(-1)
		right_output = sequence_output * right_mask.unsqueeze(-1)
		# print(entity_output)

		# Do average pooling for each mask
		left_output = left_output.sum(1) / left_mask.sum(1).unsqueeze(-1)
		entity_output = entity_output.sum(1) / entity_mask.sum(1).unsqueeze(-1)
		right_output = right_output.sum(1) / right_mask.sum(1).unsqueeze(-1)
		# print(entity_output)
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
		with torch.autograd.set_detect_anomaly(True):
			if labels is not None:
				loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
				outputs = (loss, logits) + outputs[2:]
			else:
				outputs = (logits,) + outputs[2:]

		return outputs  # (loss), logits, (hidden_states), (attentions)
