import json
import random
import os

TOTAL_NUM_LINES = 4047079
RATIOS = {'train': 0.8, 'dev': 0.1, 'test': 0.1}
SPLITS = ['train', 'dev', 'test']
ofps = {x: open(f'../json_data/{x}.json', 'w', encoding='utf8') for x in SPLITS}

idxes = list(range(TOTAL_NUM_LINES))
random.shuffle(idxes)
idx2split = {}

split_offset = 0
for split in SPLITS:
	split_size = int(TOTAL_NUM_LINES * RATIOS[split])+1
	print(f"Split {split} size: {split_size}")
	split_idxes = idxes[split_offset:split_offset+split_size]
	split_offset += split_size
	for idx in split_idxes:
		idx2split[idx] = split

for i in range(TOTAL_NUM_LINES):
	assert i in idx2split, f"i: {i} not in idx2split."

input_fn = '../json_data/all.json'
resulting_lines = {x: 0 for x in SPLITS}

with open(input_fn, 'r', encoding='utf8') as f:
	lidx = 0
	for lidx, line in enumerate(f):
		if lidx % 10000 == 0:
			print(lidx)
		json_item = json.loads(line)
		split = idx2split[lidx]
		out_line = json.dumps(json_item, ensure_ascii=False)
		ofps[split].write(out_line+'\n')
		resulting_lines[split] += 1
	print(f"lidx: {lidx}; TOTAL_NUM_LINES: {TOTAL_NUM_LINES}")

for split in SPLITS:
	ofps[split].close()

print(resulting_lines)
