import h5py

hpy = h5py.File('train_wegtypes.json_bert-base-uncased_entgraph_labels_0.0.cache.h5', 'r')

print(hpy['size'][0])