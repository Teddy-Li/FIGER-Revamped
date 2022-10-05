from torch.utils.data import Dataset, DataLoader
import torch
from typing import List, Dict, Any
from torch.nn.utils.rnn import pad_sequence
import json
import h5py
import numpy as np
# import numpy as np


class CustomError(Exception):
	pass


def tokenize_tokens(tokens, start, end, tokenizer, max_len):
	def pad_to_maxlen(seq):
		return seq + [0] * (max_len - len(seq))

	sent = ' '.join(tokens)
	ret = tokenizer(sent, max_length=max_len, truncation=True)
	bert_per_token = [(x, tokenizer(x, add_special_tokens=False)['input_ids']) for x in tokens]
	net_tok_cnt = 0
	for itok_idx in range(len(bert_per_token)):
		itok, btok = bert_per_token[itok_idx]
		if net_tok_cnt + len(btok) > max_len:
			bert_per_token[itok_idx] = (itok, btok[:max_len - net_tok_cnt])
			net_tok_cnt = max_len
		else:
			net_tok_cnt += len(btok)
	bert_tokens_net = sum([x[1] for x in bert_per_token], [])
	if len(bert_tokens_net) > 2.5*len(tokens):
		raise CustomError(f'Too many tokens: # {sent} #')
	bert_tokens_all = ret['input_ids']
	assert len(bert_tokens_all) == len(bert_tokens_net) + 2
	assert all([x == y for x, y in zip(bert_tokens_net, bert_tokens_all[1:-1])])

	mapping = {}
	left_mask = [0] * len(bert_tokens_all)
	right_mask = [0] * len(bert_tokens_all)
	entity_mask = [0] * len(bert_tokens_all)
	bt_offset = 1  # 0 is [CLS]
	ti = 0
	for ti, t in enumerate(tokens):
		mapping[ti] = list(range(bt_offset, bt_offset + len(bert_per_token[ti][1])))
		bt_offset += len(bert_per_token[ti][1])
		if ti < start:
			for i in mapping[ti]:
				left_mask[i] = 1
		elif ti < end:
			for i in mapping[ti]:
				entity_mask[i] = 1
		else:
			for i in mapping[ti]:
				right_mask[i] = 1
	assert bt_offset == len(bert_tokens_all) - 1
	assert ti == len(bert_per_token) - 1
	assert [a+b+c for a, b, c in zip(left_mask, entity_mask, right_mask)] == [0] + [1] * len(bert_tokens_all[1:-1]) + [0], \
		f"{[a+b+c for a, b, c in zip(left_mask, entity_mask, right_mask)]} != {[0] + [1] * len(bert_tokens_all[1:-1]) + [0]}"
	return pad_to_maxlen(bert_tokens_all), pad_to_maxlen(left_mask), pad_to_maxlen(entity_mask), \
		   pad_to_maxlen(right_mask), mapping


class FigerDataset(Dataset):
	def __init__(self, input_fn: str, typeset_fn: str, tokenizer, num_labels: int, smooth_labels: float = 0.0,
				 max_len: int = 128, labels_key: str = 'entgraph_labels', reload_data: bool = False,
				 debug: bool = False, cache_mode: str = 'hdf5'):
		self.input_fn = input_fn
		self.data = [] if cache_mode == 'memory' else None
		self.cache_mode = cache_mode
		self.debug = debug
		self.hpy = None
		self.entry_cnt = 0

		tokenizer_identifier = tokenizer.name_or_path.split('/')[-1]
		self.cache_fn = input_fn + f'_{tokenizer_identifier}_{labels_key}_%.1f.cache' % smooth_labels
		self.hpy_fn = self.cache_fn + '.h5'
		try:
			if reload_data is True:  # force reload
				raise FileNotFoundError

			if cache_mode == 'hdf5':
				self.hpy = h5py.File(self.hpy_fn, 'r')
			elif cache_mode == 'memory':
				with open(self.cache_fn, 'r') as ifp:
					print(f"Loading cached data from {self.cache_fn}")
					for lidx, line in enumerate(ifp):
						if lidx > 1000 and self.debug:
							break
						if lidx % 10000 == 0:
							print(f"Loading line {lidx}")
						if len(line) < 2:
							continue
						self.data.append(json.loads(line))
			else:
				raise NotImplementedError
		except FileNotFoundError:
			print(f"{self.cache_fn} not found, creating new cache from {input_fn}, this may take a while...")
			num_entries_skipped_for_not_well_formed = 0
			num_entries_hitting_max_len = 0

			with open(typeset_fn, 'r', encoding='utf8') as tfp:
				self.typeset = json.load(tfp)
				self.type_mapping = {x: i for i, x in enumerate(self.typeset)}

			if cache_mode == 'hdf5':
				max_num_entries = 3400000 if 'train' in input_fn else 450000
				self.hpy = h5py.File(self.cache_fn+'.h5', 'w')
				self.hpy.create_dataset('id', (max_num_entries,), dtype='S100')
				self.hpy.create_dataset('input_ids', (max_num_entries, max_len), dtype='int32')
				self.hpy.create_dataset('left_mask', (max_num_entries, max_len), dtype='int32')
				self.hpy.create_dataset('entity_mask', (max_num_entries, max_len), dtype='int32')
				self.hpy.create_dataset('right_mask', (max_num_entries, max_len), dtype='int32')
				self.hpy.create_dataset('labels', (max_num_entries, num_labels), dtype='float32')
				self.hpy.create_dataset('entity_names', (max_num_entries,), dtype='S100')
				self.hpy.create_dataset('sentid', (max_num_entries,), dtype='S100')
				self.hpy.create_dataset('fileid', (max_num_entries,), dtype='S100')
				self.hpy.create_dataset('size', (1,), dtype='int32')
				hpy_lcnt = 0
			elif cache_mode == 'memory':
				cfp = open(self.cache_fn, 'w', encoding='utf8')
			else:
				raise ValueError(f"Unknown cache mode {cache_mode}")

			with open(input_fn, 'r', encoding='utf8') as ifp:
				for lidx, line in enumerate(ifp):
					if lidx % 10000 == 0:
						print(f"Loading line: {lidx}; Skipped: {num_entries_skipped_for_not_well_formed}; "
							  f"hitting max len: {num_entries_hitting_max_len}")
					json_item = json.loads(line)
					try:
						bert_tokens, left_mask, entity_mask, right_mask, mapping = \
							tokenize_tokens(json_item['tokens'], json_item['start'], json_item['end'], tokenizer, max_len)
					except CustomError as e:
						print(f"Skipping line {lidx} due to {e}")
						num_entries_skipped_for_not_well_formed += 1
						continue
					if len(bert_tokens) == max_len:
						num_entries_hitting_max_len += 1
					labels = [0] * num_labels
					for lbl in json_item[labels_key]:
						labels[self.type_mapping[lbl]] = 1
					if smooth_labels > 0:
						total_sum = sum(labels)
						assert total_sum == len(json_item[labels_key])
						smoothed_zero = smooth_labels*total_sum/num_labels
						smoothed_one = 1 - smooth_labels + smooth_labels*total_sum/num_labels
						for lblidx in range(num_labels):
							if labels[lblidx] == 0:
								labels[lblidx] = smoothed_zero
							elif labels[lblidx] == 1:
								labels[lblidx] = smoothed_one
							else:
								raise AssertionError
						assert sum(labels) == len(json_item[labels_key])  # check that we didn't mess up the sum

					out_item = {
						'id': json_item['id'],
						'input_ids': bert_tokens,
						'left_mask': left_mask,
						'entity_mask': entity_mask,
						'right_mask': right_mask,
						'labels': labels,
						'entity_name': json_item['entity_name'],
						'sentid': json_item['sentid'],
						'fileid': json_item['fileid'],
					}

					if cache_mode == 'hdf5':
						self.hpy['id'][hpy_lcnt] = out_item['id']
						self.hpy['input_ids'][hpy_lcnt] = out_item['input_ids']
						self.hpy['left_mask'][hpy_lcnt] = out_item['left_mask']
						self.hpy['entity_mask'][hpy_lcnt] = out_item['entity_mask']
						self.hpy['right_mask'][hpy_lcnt] = out_item['right_mask']
						self.hpy['labels'][hpy_lcnt] = out_item['labels']
						self.hpy['entity_names'][hpy_lcnt] = out_item['entity_name'].encode('ascii', 'replace').decode('ascii')
						self.hpy['sentid'][hpy_lcnt] = out_item['sentid']
						self.hpy['fileid'][hpy_lcnt] = out_item['fileid']
						hpy_lcnt += 1
					elif cache_mode == 'memory':
						self.data.append(out_item)
						out_line = json.dumps(out_item, ensure_ascii=False)
						cfp.write(out_line + '\n')
					else:
						raise ValueError(f"Unknown cache mode {cache_mode}")
			print(f"Skipped {num_entries_skipped_for_not_well_formed} entries due to not well formed input;"
				  f" {num_entries_hitting_max_len} entries hitting max len.")
			if cache_mode == 'hdf5':
				self.hpy['size'][0] = hpy_lcnt
				self.hpy.close()
				self.hpy = h5py.File(self.hpy_fn, 'r')
			elif cache_mode == 'memory':
				cfp.close()
			else:
				raise ValueError(f"Unknown cache mode {cache_mode}")
		# print(f"Loaded {len(self.data)} entries from {input_fn}")
		length = len(self.data) if cache_mode == 'memory' else self.hpy['size'][0]
		print(f"Loaded {length} entries from {input_fn}")

	def __len__(self):
		# return len(self.data)
		return self.hpy['size'][0] if self.cache_mode == 'hdf5' else len(self.data)

	def __getitem__(self, idx):
		if self.cache_mode == 'hdf5':
			return {
				'id': self.hpy['id'][idx],
				'input_ids': self.hpy['input_ids'][idx],
				'left_mask': self.hpy['left_mask'][idx],
				'entity_mask': self.hpy['entity_mask'][idx],
				'right_mask': self.hpy['right_mask'][idx],
				'labels': self.hpy['labels'][idx],
				'entity_name': self.hpy['entity_names'][idx],
				'sentid': self.hpy['sentid'][idx],
				'fileid': self.hpy['fileid'][idx],
			}
		elif self.cache_mode == 'memory':
			return self.data[idx]
		else:
			raise ValueError(f"Unknown cache mode {self.cache_mode}")


def data_collator(features: list) -> Dict[str, Any]:
	input_ids = pad_sequence([torch.tensor(f['input_ids'], dtype=torch.long) for f in features], batch_first=True,
							 padding_value=0)
	left_mask = pad_sequence([torch.tensor(f['left_mask'], dtype=torch.long) for f in features], batch_first=True,
							 padding_value=0)
	entity_mask = pad_sequence([torch.tensor(f['entity_mask'], dtype=torch.long) for f in features], batch_first=True,
							   padding_value=0)
	right_mask = pad_sequence([torch.tensor(f['right_mask'], dtype=torch.long) for f in features], batch_first=True,
							  padding_value=0)
	labels = torch.tensor(np.array([f['labels'] for f in features]), dtype=torch.float)
	return {
		'input_ids': input_ids,
		'left_mask': left_mask,
		'entity_mask': entity_mask,
		'right_mask': right_mask,
		'labels': labels,
	}


def get_dataloader(input_fn: str, typeset_fn: str, num_labels: int, tokenizer, batch_size: int = 32,
				   max_len: int = 128, shuffle: bool = False, smooth_labels: float = 0.0,
				   labels_key: str = 'entgraph_labels', reload_data: bool = False):
	dataset = FigerDataset(input_fn, typeset_fn, tokenizer, num_labels, smooth_labels, max_len, labels_key,
						   reload_data=reload_data)
	return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
