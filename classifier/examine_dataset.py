import h5py
import json
from transformers import AutoTokenizer
from data import FigerDataset

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
with open('../raw_data/figer2entgraph_typemap.json.set', 'r') as ifp:
    typeset = json.load(ifp)

dataset = FigerDataset('../json_data/test_wegtypes.json',
                       '../raw_data/figer2entgraph_typemap.json.set',
                       tokenizer, len(typeset), 0.0, 128, 'entgraph_labels', False)

for entry in dataset:
    out_item = {
        'id': entry['id'].decode('ascii'),
        'sentid': entry['sentid'].decode('ascii'),
        'fileid': entry['fileid'].decode('ascii'),
        'entity_name': entry['entity_name'].decode('ascii'),
    }
    print(out_item)
    break

