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


def tokenize_tokens(tokens, start, end, tokenizer, max_len, end_inclusive: bool, force_encode: bool):
	"""

	:param force_encode:
	:param tokens: a list of tokens
	:param start: the start of the span (inclusive)
	:param end: the end of the span (changed to be inclusive)
	:param tokenizer:
	:param max_len:
	:param end_inclusive: whether the end token is included in the span
	:return:
	"""

	end = end if end_inclusive else end-1

	def pad_to_maxlen(seq):
		return seq + [0] * (max_len - len(seq))

	def truncate_to_maxlen(bert_toks):

		curr_len = sum(len(x[1]) for x in bert_toks[start:end+1]) if (start is not None and end is not None) else 0

		if curr_len >= max_len-2:  # if the length already exceeds maximum, we move inwards.
			assert start is not None and end is not None
			right_end = end+1
			left_end = start
			while right_end > left_end:
				if right_end - left_end == 1:
					break
				if curr_len >= max_len-2:
					right_end -= 1
					curr_len -= len(bert_toks[right_end-1][1])
				else:
					break
			return left_end, right_end

		# Otherwise, we move outwards.
		if start is not None and end is not None:
			left_end = start
			right_end = end+1
		else:
			assert start is None and end is None
			left_end = 0
			right_end = 1
		while True:
			if left_end == 0 and right_end == len(bert_toks):
				break
			if left_end == 0:
				new_left_end = left_end
				new_right_end = right_end + 1
			elif right_end == len(bert_toks):
				new_left_end = left_end - 1
				new_right_end = right_end
			else:
				new_left_end = left_end - 1
				new_right_end = right_end + 1
			curr_len = sum(len(x[1]) for x in bert_toks[new_left_end:new_right_end])
			if curr_len >= max_len-2:
				break
			else:
				left_end = new_left_end
				right_end = new_right_end
		return left_end, right_end

	bert_per_token = [(x, tokenizer(x, add_special_tokens=False)['input_ids']) for x in tokens]
	tok_error_flag = False
	for t, btks in bert_per_token:
		if btks is None or len(btks) == 0:
			tok_error_flag = True
			# print(bert_per_token)
	if tok_error_flag:
		print(f"Tokenization error: empty token!")

	if sum(len(x[1]) for x in bert_per_token) >= max_len-2:
		left_end, right_end = truncate_to_maxlen(bert_per_token)
		bert_per_token = bert_per_token[left_end:right_end]
		tokens = tokens[left_end:right_end]
		start = start-left_end if start is not None else None
		end = min(end-left_end, right_end-left_end-1) if end is not None else None  # end is still inclusive here!
	else:
		pass

	sent = ' '.join(tokens)
	ret = tokenizer(sent, max_length=max_len, truncation=True)

	net_tok_cnt = 0
	for itok_idx in range(len(bert_per_token)):
		itok, btok = bert_per_token[itok_idx]
		if net_tok_cnt + len(btok) > max_len:
			bert_per_token[itok_idx] = (itok, btok[:max_len - net_tok_cnt])
			net_tok_cnt = max_len
		else:
			net_tok_cnt += len(btok)
	bert_tokens_net = sum([x[1] for x in bert_per_token], [])
	if len(bert_tokens_net) > 2.5*len(tokens) and (not force_encode):
		raise CustomError(f'Too many tokens: # {sent} #')
	bert_tokens_all = ret['input_ids']
	assert len(bert_tokens_all) == len(bert_tokens_net) + 2, f'bert_tokens_all = {bert_tokens_all}, bert_tokens_net = {bert_tokens_net}'
	assert all([x == y for x, y in zip(bert_tokens_net, bert_tokens_all[1:-1])])

	mapping = {}
	left_mask = [0] * len(bert_tokens_all)
	right_mask = [0] * len(bert_tokens_all)
	entity_mask = [0] * len(bert_tokens_all)
	bt_offset = 1  # 0 is [CLS]
	ti = 0

	if start is None or end is None:
		# When the entity is not identified in the sentence, we use the [CLS] token as the entity token, and take
		# whole sentences as left / right contexts, since we don't know where the entity is.
		assert start is None and end is None
		entity_mask[0] = 1
		left_mask = [0] + [1] * (len(bert_tokens_all)-1)
		right_mask = [0] + [1] * (len(bert_tokens_all)-1)
	else:
		for ti, t in enumerate(tokens):
			mapping[ti] = list(range(bt_offset, bt_offset + len(bert_per_token[ti][1])))
			bt_offset += len(bert_per_token[ti][1])
			if ti < start:  # start is inclusive
				for i in mapping[ti]:
					left_mask[i] = 1
			elif ti <= end:  # end is inclusive
				for i in mapping[ti]:
					entity_mask[i] = 1
			else:
				for i in mapping[ti]:
					right_mask[i] = 1
		assert bt_offset == len(bert_tokens_all) - 1
		assert ti == len(bert_per_token) - 1
		assert sum(entity_mask) > 0, f"entity_mask = {entity_mask}, start = {start}, end = {end}, len tokens = {len(tokens)}"
		assert [a+b+c for a, b, c in zip(left_mask, entity_mask, right_mask)] == [0] + [1] * len(bert_tokens_all[1:-1]) + [0], \
			f"{[a+b+c for a, b, c in zip(left_mask, entity_mask, right_mask)]} != {[0] + [1] * len(bert_tokens_all[1:-1]) + [0]}"
	return pad_to_maxlen(bert_tokens_all), pad_to_maxlen(left_mask), pad_to_maxlen(entity_mask), \
		   pad_to_maxlen(right_mask), mapping


class FigerDataset(Dataset):
	def __init__(self, input_fn: str, typeset_fn: str, tokenizer, num_labels: int, smooth_labels: float = 0.0,
				 max_len: int = 128, labels_key: str = 'entgraph_labels', reload_data: bool = False,
				 debug: bool = False, cache_mode: str = 'hdf5', with_labels: bool = True, half_data: str = None,
				 spanend_inclusive: bool = False, force_encode: bool = False):
		self.input_fn = input_fn
		self.data = [] if cache_mode == 'memory' else None
		self.cache_mode = cache_mode
		self.debug = debug
		self.hpy = None
		self.entry_cnt = 0
		assert half_data is None or half_data in ['first', 'second']
		self.half_data = half_data
		print(f"spanend inclusive = {spanend_inclusive}")

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
				if 'train' in input_fn:
					max_num_entries = 3400000
				elif 'newsspike' in input_fn:
					max_num_entries = 8000000
				elif 'newscrawl' in input_fn:
					max_num_entries = 52815000
				else:
					max_num_entries = 450000
				self.hpy = h5py.File(self.cache_fn+'.h5', 'w')
				self.hpy.create_dataset('id', (max_num_entries,), dtype='S100')
				self.hpy.create_dataset('input_ids', (max_num_entries, max_len), dtype='int32')
				self.hpy.create_dataset('left_mask', (max_num_entries, max_len), dtype='int32')
				self.hpy.create_dataset('entity_mask', (max_num_entries, max_len), dtype='int32')
				self.hpy.create_dataset('right_mask', (max_num_entries, max_len), dtype='int32')
				if with_labels:
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
							tokenize_tokens(json_item['tokens'], json_item['start'], json_item['end'], tokenizer,
											max_len, end_inclusive=spanend_inclusive, force_encode=force_encode)
					except CustomError as e:
						print(f"Skipping line {lidx} due to {e}")
						num_entries_skipped_for_not_well_formed += 1
						continue
					if len(bert_tokens) == max_len:
						num_entries_hitting_max_len += 1

					if json_item[labels_key] is not None:
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
							assert abs(sum(labels) - len(json_item[labels_key])) < 0.00001, f"sum labels: {sum(labels)}; num keys: {len(json_item[labels_key])}"  # check that we didn't mess up the sum
					else:
						assert not with_labels
						labels = None

					if 'sentid' in json_item:
						sentid = json_item['sentid']
					else:
						assert 'lineid' in json_item and 'subsent_id' in json_item
						sentid = f"{json_item['lineid']}_{json_item['subsent_id']}"

					out_item = {
						'id': json_item['id'],
						'input_ids': bert_tokens,
						'left_mask': left_mask,
						'entity_mask': entity_mask,
						'right_mask': right_mask,
						'entity_name': json_item['entity_name'],
						'sentid': sentid,
						'fileid': json_item['fileid'],
					}
					if labels is not None:
						out_item['labels'] = labels

					if cache_mode == 'hdf5':
						self.hpy['id'][hpy_lcnt] = out_item['id']
						self.hpy['input_ids'][hpy_lcnt] = out_item['input_ids']
						self.hpy['left_mask'][hpy_lcnt] = out_item['left_mask']
						self.hpy['entity_mask'][hpy_lcnt] = out_item['entity_mask']
						self.hpy['right_mask'][hpy_lcnt] = out_item['right_mask']
						self.hpy['entity_names'][hpy_lcnt] = out_item['entity_name'].encode('ascii', 'replace').decode('ascii')
						self.hpy['sentid'][hpy_lcnt] = out_item['sentid']
						self.hpy['fileid'][hpy_lcnt] = out_item['fileid']

						if labels is not None:
							self.hpy['labels'][hpy_lcnt] = out_item['labels']

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

		full_len = self.hpy['size'][0] if self.cache_mode == 'hdf5' else len(self.data)
		if self.half_data is None or self.half_data == 'first':
			self.idx_offset = 0
		elif self.half_data == 'second':
			self.idx_offset = full_len // 2
		else:
			raise ValueError(f"Unknown half_data value {self.half_data}")

	def __len__(self):
		# return len(self.data)
		full_len = self.hpy['size'][0] if self.cache_mode == 'hdf5' else len(self.data)
		if self.half_data is None:
			return full_len
		elif self.half_data == 'first':
			return full_len // 2
		elif self.half_data == 'second':
			return full_len - full_len // 2
		else:
			raise ValueError(f"Unknown half_data value {self.half_data}")

	def __getitem__(self, idx):
		effective_idx = idx + self.idx_offset

		if self.cache_mode == 'hdf5':
			out_item = {
				'id': self.hpy['id'][effective_idx],
				'input_ids': self.hpy['input_ids'][effective_idx],
				'left_mask': self.hpy['left_mask'][effective_idx],
				'entity_mask': self.hpy['entity_mask'][effective_idx],
				'right_mask': self.hpy['right_mask'][effective_idx],
				'entity_name': self.hpy['entity_names'][effective_idx],
				'sentid': self.hpy['sentid'][effective_idx],
				'fileid': self.hpy['fileid'][effective_idx],
			}

			if 'labels' in self.hpy:
				out_item['labels'] = self.hpy['labels'][effective_idx]
			return out_item
		elif self.cache_mode == 'memory':
			return self.data[effective_idx]
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
	ids = [f['id'] for f in features]
	entity_names = [f['entity_name'] for f in features]
	sentids = [f['sentid'] for f in features]
	fileids = [f['fileid'] for f in features]

	out_item = {
		'input_ids': input_ids,
		'left_mask': left_mask,
		'entity_mask': entity_mask,
		'right_mask': right_mask,
		'id': ids,
		'entity_name': entity_names,
		'sentid': sentids,
		'fileid': fileids,
	}

	if 'labels' in features[0]:
		labels = torch.tensor(np.array([f['labels'] for f in features]), dtype=torch.float)
		out_item['labels'] = labels

	return out_item


def get_dataloader(input_fn: str, typeset_fn: str, num_labels: int, tokenizer, batch_size: int = 32,
				   max_len: int = 128, shuffle: bool = False, smooth_labels: float = 0.0,
				   labels_key: str = 'entgraph_labels', reload_data: bool = False, debug: bool = False,
				   cache_mode: str = 'hdf5', with_labels: bool = True, num_workers: int = 0,
				   spanend_inclusive: bool = False, force_encode: bool = False) -> DataLoader:
	dataset = FigerDataset(input_fn, typeset_fn, tokenizer, num_labels, smooth_labels, max_len, labels_key,
						   reload_data=reload_data, debug=debug, cache_mode=cache_mode, with_labels=with_labels,
						   spanend_inclusive=spanend_inclusive, force_encode=force_encode)
	return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=data_collator)
