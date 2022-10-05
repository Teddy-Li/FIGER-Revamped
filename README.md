# Figer Classifier

### Data Pre-processing

1. Download the data from [here](https://drive.google.com/open?id=0B52yRXcdpG6MMnRNV3dTdGdYQ2M), and unzip it to the `raw_data` directory;
2. in `data_proc`, run `python load_figer_data.py` to generate the json data files (will be stored in `json_data` directory);
3. in `data_proc`, run `python data_split.py` to split the data into train/dev/test sets;
4. in `data_proc`, run `python create_label_mapping.py --task map2figer --subset [train / dev / test / all]` to create corresponding subsets with FIGER labels;
5. in `data_proc`, run `python create_label_mapping.py --task assignegtype --subset [train / dev / test / all]` to create corresponding subsets with the desired label set for entGraph entity typing;

### Type Ontology Construction

The above pre-processing process depends on the type ontology. This includes the following files:
- amended_types.map: the amended mapping from Freebase labels to FIGER labels;
- weaker_types_map.tsv: the 'weaker' mapping from Freebase labels to FIGER labels we additionally added;
- figer2entgraph_type_map.json: the mapping from FIGER labels to entGraph labels;
- figer2entgraph_type_map.json.set: the set of entGraph labels;

The procedure is to first map the Freebase labels in the output of `data_split.py` to the original FIGER labels, 
then map these FIGER labels to the desired labels for entGraph entity typing.

To amend the FIGER ontology and create the final mapping, we start from the `SUBSET.json` files and do the followings:
1. Fetch `types.map` from [here](https://github.com/xiaoling/figer/blob/master/src/main/resources/edu/washington/cs/figer/analysis/types.map),
put it under `raw_data` directory as `original_types.map`, copy it to `amended_types.map`, and create an empty tsv file `weaker_types_map.tsv`;
2. in `data_proc`, run `python create_label_mapping.py --task check --check_fn XXX` to collect information about the distribution of 
Freebase labels in the corresponding file, look for popular Freebase labels not covered by the original `types.map`, and add them to `amended_types.map`;
there are some Freebase labels which can be mapped to some FIGER labels when no other mapping is available, 
but are unsafe to be mapped to those FIGER labels when other options exist, these are added to `weaker_types_map.tsv`;
3. Also check for incorrect mappings from the original `types.map`, remove them from `amended_types.map`, and add them to `removed_types.map`;
4. create `SUBSET_wfiger.json` files by running `python create_label_mapping.py --task map2figer --subset XXX`, 
then re-run `check` task in `create_label_mapping.py` to check for any remaining Freebase labels not covered by the amended `types.map`;
5. Repeat the above process until satisfactory recall is met (~ 95%), then calculate the relative importance of each 
FIGER label by running `python create_label_mapping.py --task sort --subset all`; importance is defined as a weighted sum of the number of occurrences, 
where the weights are reciprocal of the number of FIGER labels in each entry.
6. According to the relative importance, merge the less important FIGER labels into the more important ones, 
and split the over-popular FIGER labels into sub-categories to avoid unmanageably-large sub-graphs; in this process, 
one creates a further mapping from the two-layer FIGER labels to a one-layer entGraph label set, parallel to the FIGER 
first-layer labels from previous EG experiments.
7. The mapping from FIGER labels to entGraph labels is specified in the function `generate_entgraph_typeset` in `create_label_mapping.py`,
and the creation of the mapping / typeset files is done by running `python create_label_mapping.py --task generate_entgraph_typemap --subset all`;

At this point the dependencies for the above pre-processing steps are satisfied, 
the output of data pre-processing will be used as training data in the section below.

### Developing Classifier
#### Overview
The classifier is a multi-class multi-label sequence classifier based on `bert-base-uncased`.
There are two major design choices: which tokens to extract representations from, and how large a language model to use;
- Representation extraction:
  1. use the \[CLS\] token;
  2. use the average of entity tokens;
  3. concatenate `i` and `ii`;
  4. use the average of left context and average of right context tokens (concatenated);
  5. concatenate `ii` and `iv`.
- How large a model: given a fixed computation constraint, we can choose to use a larger model with fp16,
or a smaller model with fp32. The hypothesis is that using a larger model with fp16 will give better performance.
- Hyper-parameters: other hyper-parameters of interest include: learning rate, number of epochs, 
metric for best model (\[macro_f1, micro_f1\]).

#### Steps
1. Do cache: `python train.py --do_cache`, this will cache the representations of data entries;
2. Do train: @ ./classifier;
    - On private servers: do `nohup python -u train.py --do_train --model_name_or_path ../../lms/bert-base-uncased --encode_mode entity --lr 5e-5 --num_train_epochs 5 --metric_best_model macro_f1 --label_smoothing_factor 0.0 > ./logdir/bbu_entity_lsf0.0_5e-5.log &`
    - On Cluster: do `sbatch -p ILCC_GPU --gres gpu:4 -o ./logdir/bbu_cls_entity_lsf0.0_5e-5.log train_script.sh bert-base-uncased cls_entity 5e-5 0.0 /disk/scratch/tli/figer_simple_classifier/model_ckpts/json_data/ /disk/scratch/tli/figer_simple_classifier/model_ckpts/ ../model_ckpts`;
    - Key tunable hyperparameters include:
      - `--encode_mode`: which tokens' representation to take for the classifier;
      - `--typeset_fn` / `--labels_key`: which typeset to use;
      - `--lr`: the learning rate;
      - `--use_fp16`: whether to use fp16 for speeding up;
      - `--metric_best_model`: which metric to use for selecting the best model, by default macro_f1, values from different metrics should be roughly aligned;
      - `--reload_data`: whether to reload the data (refresh cache), needed when data set is updated;
      - `--label_smoothing_factor`: the label smoothing factor, by default 0.0, values from 0.0 to 0.2 are reasonable;
3. Do eval: @ ./classifier:
   - On private servers, do: ``;
   - On Cluster, do: ``;

### Doing Inference for NewsSpike / NewsCrawl
To be specified.