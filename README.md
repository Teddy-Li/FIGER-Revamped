# Figer Classifier
This repository contains the code for revamping the FIGER entity typing ontology and dataset, and a simple classifier on the revamped dataset based on [BERT-base-uncased](https://huggingface.co/bert-base-uncased).

With the revamped ontology, we are able to achieve a 94.3% macro-F1 score on the test set, XXX improvement over the original FIGER dataset.

| Model               | Dataset        | Macro-F1 | Micro-F1 |
|---------------------|----------------|----------|----------|
| BERT-base-uncased   | Original FIGER | XXXX     | XXXX     |
| BERT-base-uncased   | Revamped FIGER | 94.3     | 95.1     |
 | BERT-large-uncased  | Original FIGER | XXXX     | XXXX     |
 | BERT-large-uncased  | Revamped FIGER | XXXX     | XXXX     |


We use this classifier to type the entities for our [Entailment Graph project](), stay tuned for more details.

## Developing Classifier
### Overview
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
- Note: empirically we have found using `entity` tokens for representation to be most efficient; removing entity representations harm the performance dramatically, whereas additionally including left and right contexts are not clearly better.

### Steps

0. NOTE: if you are using the slurm scripts, please change the `#SBATCH` options to match your environment (see [this](https://slurm.schedmd.com/sbatch.html) for detailed documentation);
1. Do cache: `python train.py --do_cache --label_smoothing_factor 0.1 --reload_data`, this will cache the representations of data entries;
2. Do train: @ ./classifier;
    - On private servers: do `nohup python -u train.py --do_train --model_name_or_path ../../lms/bert-base-uncased --encode_mode entity --lr 5e-5 --num_train_epochs 5 --metric_best_model macro_f1 --label_smoothing_factor 0.0 > ./logdir/bbu_entity_lsf0.0_5e-5.log &`
    - On Cluster: do `sbatch -p ILCC_GPU --gres gpu:4 -o ./logdir/bbu_cls_entity_lsf0.0_5e-5.log train_script.sh bert-base-uncased cls_entity 5e-5 0.0 /disk/scratch/tli/figer_simple_classifier/model_ckpts/json_data/ /disk/scratch/tli/figer_simple_classifier/model_ckpts/ ../model_ckpts`;
    - Key tunable hyperparameters include:
      - `--encode_mode`: which tokens' representation to take for the classifier;
      - `--num_clsf_layers`: how many layer MLP to use for the classifier;
      - `--typeset_fn` / `--labels_key`: which typeset to use;
      - `--lr`: the learning rate;
      - `--use_fp16`: whether to use fp16 for speeding up;
      - `--metric_best_model`: which metric to use for selecting the best model, by default macro_f1, values from different metrics should be roughly aligned;
      - `--reload_data`: whether to reload the data (refresh cache), needed when data set is updated;
      - `--label_smoothing_factor`: the label smoothing factor, by default 0.0, values from 0.0 to 0.2 are reasonable;
3. Do eval: @ ./classifier:
   - On private servers, do: ``;
   - On Cluster, do: `sbatch -p ILCC_GPU --gres gpu:2 -o ./logdir/bbu_entity_lsf0.0_5e-5_dev.log eval_script.sh bert-base-uncased entity /home/s2063487/figer_simple_classifier/model_ckpts/model_ckpts/bert-base-uncased/entity_5e-05_0.0/checkpoint-95000 /tli/figer_simple_classifier/model_ckpts/json_data/ dev`;
4. Test predict: @ ./classifier:
   - On cluster, do: `sbatch -p ILCC_GPU -w duflo --gres gpu:2 -o ./logdir/predict_test.log predict_script.sh bert-base-uncased entity /home/s2063487/figer_simple_classifier/model_ckpts/model_ckpts/bert-base-uncased/entity_5e-05_0.0/checkpoint-95000 /tli/figer_simple_classifier/model_ckpts/json_data/ test_wegtypes _bert-base-uncased_entgraph_labels_0.0 0.05 --is_inference --debug`;

## Doing Inference for NewsSpike / NewsCrawl
Note: you can create your own inference scripts analogous to `news_proc` and `levy_proc`.

1. load_news_data:
    - *newsspike*: @news_proc `nohup python -u load_news_data.py --in_path ../../entGraph/news_gen8_p.json --out_dir ../news_data/ --data_name newsspike --out_fn %s_gparser_typing_input.json --mode load > ns_load.log &`;
    - *newscrawl*: @news_proc `nohup python -u load_news_data.py --in_path ../../news_genC_GG.json --out_dir ../news_data/ --data_name newscrawl --out_fn %s_gparser_typing_input.json --mode load > nc_load.log &`;
    - *LevyHolt*: @levy_proc `python -u load_levy_data.py`
2. split loaded news data:
   - *newsspike*: `nohup python -u load_news_data.py --out_dir ../news_data/ --data_name newsspike --out_fn %s_gparser_typing_input.json --mode split --num_slices 8 --expected_num_lines 63876006 > ns_split.log &`;
   - *newscrawl*: `nohup python -u load_news_data.py --out_dir ../news_data/ --data_name newscrawl --out_fn %s_gparser_typing_input.json --mode split --num_slices 120 --expected_num_lines 1584274524 > nc_split.log &`;
   - *LevyHolt*: NA;

[//]: # (3. do cache:)

[//]: # (   - *newsspike*: `sbatch -p ILCC_CPU -o ./logdir/cache_ns_%a_%A.log --array 3-4%4 cache_script.sh bert-base-uncased ../news_data/newsspike_gparser_typing_input 0.0 --reload_data`;)

[//]: # (   - *newscrawl*: `sbatch -p ILCC_CPU -o ./logdir/cache_nc_%a_%A.log --array 0-5%3 cache_script.sh bert-base-uncased ../news_data/newscrawl_gparser_typing_input 0.0 --reload_data`;)

[//]: # (   - *newsspike*: `nohup bash cache_script_pata.sh bert-base-uncased ../news_data/newsspike_gparser_typing_input_4.json 0.0 --reload_data > ./logdir/cache_ns_4.log &`;)
3. do cache: 
   - *newsspike*: NA;
   - *newscrawl*: NA;
   - *LevyHolt*: `nohup bash cache_script_pata.sh bert-base-uncased ../levy_data/dev_input.json 0.0 --reload_data --spanend_inclusive --force_encode > ./logdir/cache_levy_dev.log &` (the cached files can then be sent to MLP server);
4. do predict:

[//]: # (   - *newsspike*: `sbatch -p ILCC_GPU -w nuesslein --gres gpu:4 -o ./logdir/predict_ns_7.log predict_script.sh bert-base-uncased entity /home/s2063487/figer_simple_classifier/model_ckpts/model_ckpts/bert-base-uncased/entity_5e-05_0.0/checkpoint-95000 /disk/scratch/tli/figer_simple_classifier/model_ckpts/news_data/ newsspike_gparser_typing_input_7 _bert-base-uncased_entgraph_labels_0.0 0.05 --is_inference`;)
   - *newsspike*: `sbatch -p ILCC_GPU --exclude duflo --array 0 --gres gpu:4 -o ./logdir/predict_ns_%a_%A.log predict_script_array.sh bert-base-uncased entity /home/s2063487/figer_simple_classifier/model_ckpts/model_ckpts/bert-base-uncased/entity_5e-05_0.0/checkpoint-95000 /disk/scratch/tli/figer_simple_classifier/model_ckpts/news_data/ newsspike_gparser_typing_input entgraph_labels 0.05 --is_inference --spanend_inclusive`;
   - *newsspike*: `sbatch -p PGR-Standard --array 4-7 --gres gpu:2 -o ./logdir/predict_ns_%a_%A.log predict_script_array.sh bert-base-uncased entity /home/s2063487/figer_simple_classifier/model_ckpts/model_ckpts/bert-base-uncased/entity_5e-05_0.0/checkpoint-95000 /disk/scratch_big/tli/figer_simple_classifier/model_ckpts/news_data/ newsspike_gparser_typing_input entgraph_labels 0.05 --is_inference --spanend_inclusive`;
   - *newscrawl*: `sbatch -p ILCC_GPU --exclude duflo,levi --array 100-119%4 --gres gpu:4 -o ./logdir/predict_nc_%a_%A.log predict_script_array.sh bert-base-uncased entity /home/s2063487/figer_simple_classifier/model_ckpts/model_ckpts/bert-base-uncased/entity_5e-05_0.0/checkpoint-95000 /disk/scratch/tli/figer_simple_classifier/model_ckpts/news_data/ newscrawl_gparser_typing_input entgraph_labels 0.01 --is_inference --spanend_inclusive`;
   - *newscrawl*: `sbatch -p PGR-Standard --array 0-10:4 --gres gpu:2 -o ./logdir/predict_nc_%a_%A.log predict_script_array.sh bert-base-uncased entity /home/s2063487/figer_simple_classifier/model_ckpts/model_ckpts/bert-base-uncased/entity_5e-05_0.0/checkpoint-95000 /disk/scratch_big/tli/figer_simple_classifier/model_ckpts/news_data/ newscrawl_gparser_typing_input entgraph_labels 0.05 --is_inference --spanend_inclusive`;
   - *newsspike*: `python -u train.py --do_predict --data_dir ../news_data/ --predict_fn newsspike_gparser_typing_input_0.json --model_name_or_path ../../lms/bert-base-uncased --encode_mode entity --output_dir ../model_ckpts/bert-base-uncased/entity_5e-05_0.0/checkpoint-95000 --predict_threshold 0.05 --is_inference --predict_half first`;
   - *LevyHolt*: `sbatch -p PGR-Standard --gres gpu:2 -o ./logdir/predict_levy_dev.log predict_script.sh bert-base-uncased entity /home/s2063487/figer_simple_classifier/model_ckpts/model_ckpts/bert-base-uncased/entity_5e-05_0.0/checkpoint-95000 levy_data /disk/scratch_big/tli/figer_simple_classifier/model_ckpts/levy_data/ dev_input _bert-base-uncased_entgraph_labels_0.0 0.05 --is_inference --spanend_inclusive --force_encode`;

5. Integrate Results:
   - *newsspike*: `nohup python -u integrate_results.py --data_dir ../news_data/ --data_name newsspike --num_slices 8 --job_name model > ns_integrate_modelout.log &`;

6. Generate typed corpus:
   - *newsspike*: `nohup python -u integrate_results.py --data_dir ../news_data/ --data_name newsspike --output_fn %s_gparser_typing_output.json --job_name corpus --parsed_fn ../../entGraph_NS/news_gen8_p.json > ns_integrate_corpus.log &`;