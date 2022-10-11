import json
import h5py

# TOTAL_NUM_LINES = 4047079
# input_fn = '../json_data/all.json'
#
# outlier_cnt = 0
#
# with open(input_fn, 'r', encoding='utf8') as f:
# 	for lidx, line in enumerate(f):
# 		if lidx % 10000 == 0:
# 			print(f"lidx: {lidx} / {TOTAL_NUM_LINES}; outlier_cnt: {outlier_cnt}")
# 		json_item = json.loads(line)
# 		if len(json_item['tokens']) > 100:
# 			outlier_cnt += 1
#
#
# print(f"Outlier count: {outlier_cnt}")


f = h5py.File('../json_data//Users/teddy/PycharmProjects/figer_simple_classifier/json_data/test_wegtypes.json_bert-base-uncased_entgraph_labels_0.0.cache.h5', 'r')

