import json
import argparse
from typing import List

TOTAL_NUM_LINES = 4047079

FIGER_LABELS = {'art', 'astral_body', 'award', 'biology', 'body_part', 'broadcast', 'broadcast_network',
				'broadcast_program', 'building', 'chemistry', 'computer', 'disease', 'education', 'event', 'finance',
				'food', 'game', 'geography', 'god', 'government', 'government_agency', 'internet', 'language', 'law',
				'living_thing', 'livingthing', 'location', 'medicine', 'metropolitan_transit', 'military', 'music',
				'news_agency', 'newspaper', 'organization', 'park', 'people', 'person', 'play', 'product', 'rail',
				'religion', 'software', 'thing', 'time', 'title', 'train', 'transit', 'transportation', 'visual_art',
				'written_work'}


def read_existing_label_map(map_fn):
	"""
	Read existing label map from file.
	:return: dict
	"""
	if map_fn is None:
		return dict()
	label_map = {}
	with open(map_fn, 'r', encoding='utf8') as f:
		for line in f:
			line = line.strip()
			if line == '':
				continue
			lbls = line.split('\t')
			fbs_lbl = lbls[0]
			fgr_lbls = lbls[1:]
			assert len(fgr_lbls) > 0
			label_map[fbs_lbl] = fgr_lbls
	return label_map


def label_is_meaningful(labels: List[str]) -> bool:
	if len(labels) > 0 and not (len(labels) == 1 and labels[0] == '/common/topic'):
		return True
	return False


def refine_label_map(input_fn='../json_data/%s.json'):
	label_cnts = {}

	for split in ['all', 'train', 'dev', 'test']:
		with open(input_fn % split, 'r', encoding='utf8') as f:
			for lidx, line in enumerate(f):
				if lidx % 10000 == 0:
					print(f"lidx: {lidx}; split: {split}")
				json_item = json.loads(line)
				labels = json_item['labels']

				# if we already have some figer labels, we only care about those entries without a label.
				if 'figer_extended_labels' in json_item and len(json_item['figer_extended_labels']) > 0:
					continue

				for lbl in labels:
					if lbl not in label_cnts:
						label_cnts[lbl] = {'all': 0, 'train': 0, 'dev': 0, 'test': 0}
					label_cnts[lbl][split] += 1

	label_cnts = {k: v for k, v in sorted(label_cnts.items(), key=lambda item: item[1]['all'], reverse=True)}
	existing_lbl_map = read_existing_label_map('../raw_data/amended_types.map')
	weak_lbl_map = read_existing_label_map('../raw_data/weak_types_map.tsv')

	for i, (k, v) in enumerate(label_cnts.items()):
		if i >= 1000:
			break
		if k in existing_lbl_map:
			print(f"{i}: Type {k} exists in the label map.")
			continue
		elif k in weak_lbl_map:
			print(f"{i}: Type {k} exists in the weak label map.")
			continue
		else:
			print(f"{i}: {k}: {v['all']}; {v['train']}; {v['dev']}; {v['test']}")
	print(f"Total label count: {len(label_cnts)}")

# for split in ['train', 'dev', 'test']:
# 	print(f"Split: {split}")
# 	for label in label_cnts:
# 		# assert label_cnts[label][split] > 0, f"Label {label} not in split {split}"
# 		print(f"{label}: {label_cnts[label][split]}")

# labels = {x: [] for x in label_cnts}
# print(f"Total number of labels: {len(labels)}")
# with open('../json_data/raw2myfiger_typing.json', 'w', encoding='utf8') as f:
# 	json.dump(labels, f, ensure_ascii=False, indent=4)


def get_figer_labels_for_data(input_fn, output_fn, ling_map_fn, amended_map_fn, weak_map_fn):
	ling_map = read_existing_label_map(ling_map_fn)
	amended_map = read_existing_label_map(amended_map_fn)  # will be empty dict if amended_map_fn is None
	weak_map = read_existing_label_map(weak_map_fn)  # will be empty dict if weak_map_fn is None

	unmapped_label_cnt = 0
	ling_unmapped_entry_cnt = 0
	extended_unmapped_entry_cnt = 0

	with open(input_fn, 'r', encoding='utf8') as in_fp, open(output_fn, 'w', encoding='utf8') as out_fp:
		for lidx, line in enumerate(in_fp):
			if lidx % 10000 == 0:
				print(f"lidx: {lidx}; unmapped_label_cnt: {unmapped_label_cnt}; "
					  f"ling_unmapped_entry_cnt: {ling_unmapped_entry_cnt}; extended_unmapped_entry_cnt: {extended_unmapped_entry_cnt}")
			json_item = json.loads(line)
			labels = json_item['labels']
			figer_ling_labels = set()
			figer_extended_labels = set()
			figer_weak_labels = set()
			for lbl in labels:
				if lbl in ling_map:
					figer_ling_labels.update(ling_map[lbl])

				if lbl in amended_map:
					figer_extended_labels.update(amended_map[lbl])
				elif lbl in weak_map:
					figer_weak_labels.update(weak_map[lbl])
				else:
					# print(f"Label {lbl} not in any map.")
					unmapped_label_cnt += 1
			if len(figer_ling_labels) == 0 and label_is_meaningful(labels):
				ling_unmapped_entry_cnt += 1

			if len(figer_extended_labels) == 0:
				figer_extended_labels = figer_weak_labels
			if len(figer_extended_labels) == 0 and label_is_meaningful(labels):  # if after introducing the weak labels, the figer labels still is empty
				# print(f"Entry {json_item['id']} has no figer label.")
				extended_unmapped_entry_cnt += 1

			figer_ling_labels = list(figer_ling_labels)
			figer_extended_labels = list(figer_extended_labels)
			json_item['figer_ling_labels'] = figer_ling_labels
			json_item['figer_extended_labels'] = figer_extended_labels
			out_fp.write(json.dumps(json_item, ensure_ascii=False) + '\n')
	print(f"unmapped_label_cnt: {unmapped_label_cnt}; ling_unmapped_entry_cnt: {ling_unmapped_entry_cnt};"
		  f" extended_unmapped_entry_cnt: {extended_unmapped_entry_cnt}")


def sort_types_with_importance(input_fn, record_fn, label_mode='figer_extended_labels'):

	type_importance_dct = {}

	with open(input_fn, 'r', encoding='utf8') as in_fp:
		for lidx, line in enumerate(in_fp):
			if lidx % 10000 == 0:
				print(f"lidx: {lidx}")
			json_item = json.loads(line)

			# TODO: remove respective first-level labels when a more specific second-level label is considered important.

			for lbl in json_item[label_mode]:
				lbl_chain = lbl.split('/')
				assert lbl_chain[0] == ''
				lbl_chain = lbl_chain[1:]
				if lbl_chain[0] not in type_importance_dct:
					type_importance_dct[lbl_chain[0]] = dict()
				if len(lbl_chain) == 1:
					lv2_name = 'others'
				else:
					lv2_name = lbl_chain[1]
				if lv2_name not in type_importance_dct[lbl_chain[0]]:
					type_importance_dct[lbl_chain[0]][lv2_name] = {'weight': 0, 'cnt': 0}

				type_importance_dct[lbl_chain[0]][lv2_name]['cnt'] += 1
				type_importance_dct[lbl_chain[0]][lv2_name]['weight'] += 1 / len(json_item[label_mode])

	for lv1_name in type_importance_dct:
		total_cnt = 0
		total_weight = 0
		for lv2_name in type_importance_dct[lv1_name]:
			total_cnt += type_importance_dct[lv1_name][lv2_name]['cnt']
			total_weight += type_importance_dct[lv1_name][lv2_name]['weight']
		type_importance_dct[lv1_name]['total'] = {'weight': total_weight, 'cnt': total_cnt}
		type_importance_dct[lv1_name] = {k: v for k, v in sorted(type_importance_dct[lv1_name].items(), key=lambda item: item[1]['weight'], reverse=True)}
	type_importance_dct = {k: v for k, v in sorted(type_importance_dct.items(), key=lambda item: item[1]['total']['weight'], reverse=True)}

	for lv1_name in type_importance_dct:
		print(f"lv1 type: {lv1_name}; weight: {type_importance_dct[lv1_name]['total']['weight']}; cnt: {type_importance_dct[lv1_name]['total']['cnt']}")

	with open(record_fn, 'w', encoding='utf8') as out_fp:
		json.dump(type_importance_dct, out_fp, ensure_ascii=False, indent=4)


def generate_entgraph_typeset(amended_map_fn: str, weak_map_fn: str, entgraph_typemap_fn: str):
	amended_types = read_existing_label_map(amended_map_fn)
	weak_types = read_existing_label_map(weak_map_fn)

	amended_figer_set = set()
	for k, v in amended_types.items():
		amended_figer_set.update(v)
	for k, v in weak_types.items():
		amended_figer_set.update(v)

	amended_figer_set = sorted(list(amended_figer_set))

	amended_figer_entgraph_map = {x: '/'.join(x.split('/')[:2]) for x in amended_figer_set}
	amended_figer_entgraph_map['/location/city'] = '/location_admin'
	amended_figer_entgraph_map['/location/country'] = '/location_admin'
	amended_figer_entgraph_map['/location/province'] = '/location_admin'
	amended_figer_entgraph_map['/location/county'] = '/location_admin'
	amended_figer_entgraph_map['/location/admin'] = '/location_admin'
	amended_figer_entgraph_map['/person/artist'] = '/personart'
	amended_figer_entgraph_map['/person/actor'] = '/personart'
	amended_figer_entgraph_map['/person/musician'] = '/personart'
	amended_figer_entgraph_map['/person/director'] = '/personart'
	amended_figer_entgraph_map['/person/athlete'] = '/personart'
	amended_figer_entgraph_map['/person/author'] = '/personart'
	amended_figer_entgraph_map['/person/character'] = '/personmyth'
	amended_figer_entgraph_map['/person/god'] = '/personmyth'
	amended_figer_entgraph_map['/person/politician'] = '/personpol'
	amended_figer_entgraph_map['/person/noble_person'] = '/personpol'
	amended_figer_entgraph_map['/person/monarch'] = '/personpol'
	amended_figer_entgraph_map['/person/religious_leader'] = '/personpol'
	amended_figer_entgraph_map['/organization/company'] = '/corporation'
	amended_figer_entgraph_map['/organization/brand'] = '/corporation'
	amended_figer_entgraph_map['/organization/airline'] = '/corporation'

	# manually add the '/thing' label for active detection of entities falling in other categories.
	all_entgraph_types = sorted(list(set(amended_figer_entgraph_map.values())) + ['/thing'])

	with open(entgraph_typemap_fn, 'w', encoding='utf8') as out_fp:
		json.dump(amended_figer_entgraph_map, out_fp, ensure_ascii=False, indent=4)
	with open(entgraph_typemap_fn + '.set', 'w', encoding='utf8') as out_fp:
		json.dump(all_entgraph_types, out_fp, ensure_ascii=False, indent=4)


def get_entgraph_typeset_for_data(input_fn, output_fn, entgraph_typemap_fn, coarse_typemap_fn,
								  label_mode='figer_extended_labels'):
	with open(entgraph_typemap_fn, 'r', encoding='utf8') as in_fp:
		entgraph_typemap = json.load(in_fp)
	print(f"entgraph_typemap size: {len(set(entgraph_typemap.values()))}")

	with open(coarse_typemap_fn, 'r', encoding='utf8') as in_fp:
		coarse_typemap = json.load(in_fp)
	print(f"coarse_typemap size: {len(set(coarse_typemap.values()))}")

	skipped_unmeaningful_entry_cnt = 0
	lidx = 0

	with open(input_fn, 'r', encoding='utf8') as in_fp, open(output_fn, 'w', encoding='utf8') as out_fp:
		for lidx, line in enumerate(in_fp):
			if lidx % 10000 == 0:
				print(f"lidx: {lidx}; skipped_unmeaningful_entry_cnt: {skipped_unmeaningful_entry_cnt}")
			json_item = json.loads(line)
			figer_labels = json_item[label_mode]
			entgraph_labels = set()
			coarse_labels = set()
			for lbl in figer_labels:
				if label_mode == 'figer_extended_labels':
					entgraph_labels.add(entgraph_typemap[lbl])
					coarse_labels.add(coarse_typemap[lbl])
				elif label_mode == 'figer_ling_labels':
					lbl_lv1 = '/'.join(lbl.split('/')[:2])
					entgraph_labels.add(lbl_lv1)
					coarse_labels.add(lbl_lv1)
				else:
					raise ValueError(f"Unknown label mode: {label_mode}")

			assert len(entgraph_labels) > 0 or len(figer_labels) == 0
			if len(entgraph_labels) == 0:
				if label_is_meaningful(json_item['labels']) is True:
					entgraph_labels.add('/thing')
				# If the entry does not have a meaningful Freebase label assigned to it, we take it that there is not
				# enough information to assign a type to it, so we remove this entry from training data.
				else:
					skipped_unmeaningful_entry_cnt += 1
					continue
			if len(coarse_labels) == 0:
				# here the label is guaranteed to be meaningful.
				coarse_labels.add('/thing')
			json_item['entgraph_labels'] = sorted(list(entgraph_labels))
			json_item['coarse_labels'] = sorted(list(coarse_labels))
			out_fp.write(json.dumps(json_item, ensure_ascii=False) + '\n')

	print(f"Total lines: {lidx}; total skipped_unmeaningful_entry_cnt: {skipped_unmeaningful_entry_cnt}")
	print(f"Done.")


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--check_fn', type=str, default='../json_data/%s.json')

	parser.add_argument('--subset', type=str, default='train', help='train/dev/test')
	parser.add_argument('--ling_map_fn', type=str, default='../raw_data/original_types.map', help='Ling et. al. label map')
	parser.add_argument('--amended_map_fn', type=str, default='../raw_data/amended_types.map', help='Our amended label map')
	parser.add_argument('--weak_map_fn', type=str, default='../raw_data/weaker_type_map.tsv', help='Our weak label map')

	parser.add_argument('--entgraph_typemap_fn', type=str, default='../raw_data/figer2entgraph_typemap.json', help='EntGraph type map')
	parser.add_argument('--coarse_typemap_fn', type=str, default='../raw_data/figer2coarse_typemap.json', help='Coarse type map')
	parser.add_argument('--task', type=str, default='check/map/sort/level')

	args = parser.parse_args()

	if args.task == 'check':
		refine_label_map(args.check_fn)
	elif args.task == 'map2figer':
		get_figer_labels_for_data(f'../json_data/{args.subset}.json',
								  f'../json_data/{args.subset}_wfiger.json',
								  args.ling_map_fn, args.amended_map_fn, args.weak_map_fn)
	elif args.task == 'sort':
		sort_types_with_importance(f'../json_data/{args.subset}_wfiger.json', '../raw_data/type_importance_dct.json',
								   label_mode='figer_extended_labels')
	elif args.task == 'generate_entgraph_typemap':
		generate_entgraph_typeset(args.amended_map_fn, args.weak_map_fn, args.entgraph_typemap_fn)
	elif args.task == 'assignegtype':
		get_entgraph_typeset_for_data(f'../json_data/{args.subset}_wfiger.json',
									  f'../json_data/{args.subset}_wegtypes.json',
									  args.entgraph_typemap_fn, args.coarse_typemap_fn,
									  label_mode='figer_extended_labels')
	else:
		raise ValueError(f"Invalid task {args.task}")
