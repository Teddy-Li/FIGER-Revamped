from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoConfig, IntervalStrategy
import torch
from model import BertForFiger
from data import FigerDataset, data_collator, get_dataloader
from utils import compute_metrics, get_labelset
import json
import numpy as np
import os
import argparse


def train(model_name_or_path: str, types_list: list, encode_mode: str, num_clsf_layers: int, train_fn: str, dev_fn: str,
		  typeset_fn: str, output_dir: str, max_len: int = 128, per_device_train_batch_size: int = 32,
		  per_device_eval_batch_size: int = 64, num_train_epochs: int = 3, learning_rate: float = 5e-5,
		  weight_decay: float = 0.01, warmup_ratio: float = 0, gradient_accumulation_steps: int = 1,
		  save_steps: int = 500, logging_steps: int = 500, seed: int = 42, use_fp16: bool = False,
		  metric_best_model: str = 'macro_f1', label_smoothing_factor: float = None, reload_data: bool = False,
		  labels_key: str = 'entgraph_labels', debug: bool = False, force_encode: bool = False):
	num_labels = len(types_list)
	fp16_flag, fp16_opt_level, fp16_fulleval = (True, 'O1', True) if use_fp16 else (False, None, False)

	tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, do_lower_case=True)
	config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
	config.update({'encode_mode': encode_mode})
	config.update({'num_clsf_layers': num_clsf_layers})
	model = BertForFiger.from_pretrained(model_name_or_path, config=config)

	print(f"Building training set entries......")
	train_dataset = FigerDataset(train_fn, typeset_fn, tokenizer, num_labels, label_smoothing_factor, max_len, labels_key,
						   reload_data=reload_data, debug=debug, force_encode=force_encode)
	print(f"Training set length: {len(train_dataset)}")
	print(f"Building dev set entries......")
	dev_dataset = FigerDataset(dev_fn, typeset_fn, tokenizer, num_labels, label_smoothing_factor, max_len, labels_key,
							reload_data=reload_data, debug=debug, force_encode=force_encode)
	print(f"Dev set length: {len(dev_dataset)}")

	# test_dataset = FigerDataset(test_fn, typeset_fn, tokenizer, num_labels, label_smoothing_factor, max_len, labels_key,
	# 						reload_data=reload_data, debug=debug, force_encode=force_encode)

	with open(typeset_fn, 'r', encoding='utf8') as tfp:
		types_list = json.load(tfp)
		assert isinstance(types_list, list)

	# Single node DataParallel is automatically applied by Trainer!
	training_args = TrainingArguments(
		output_dir=output_dir,  # output directory
		overwrite_output_dir=False,  # overwrite the content of the output directory
		do_train=True,  # do training
		evaluation_strategy=IntervalStrategy.STEPS,  # evaluation strategy to adopt during training
		per_device_train_batch_size=per_device_train_batch_size,  # batch size per device during training
		per_device_eval_batch_size=per_device_eval_batch_size,  # batch size for evaluation
		gradient_accumulation_steps=gradient_accumulation_steps,  # number of updates steps to accumulate before performing a backward/update pass
		eval_accumulation_steps=1,  # number of steps to accumulate before evaluating
		learning_rate=learning_rate,  # learning rate
		weight_decay=weight_decay,  # strength of weight decay
		num_train_epochs=num_train_epochs,  # total # of training epochs
		lr_scheduler_type='linear',  # linear scheduler
		warmup_ratio=warmup_ratio,  # ratio of warmup steps used in the scheduler
		logging_dir='./logs',  # directory for storing TensorBoard logs
		logging_first_step=True,  # log and evaluate the first `global_step`-th training step
		logging_steps=logging_steps,  # log & save weights each logging_steps
		save_steps=save_steps,  # save checkpoint each save_steps
		save_total_limit=3,  # number of total save model checkpoints
		seed=seed,  # seed for initializing training
		fp16=fp16_flag,  # whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit
		fp16_opt_level=fp16_opt_level,  # for fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. See details at https://nvidia.github.io/apex/amp.html
		fp16_full_eval=fp16_fulleval,  # whether to use full fp16 evaluation
		dataloader_num_workers=8,  # number of subprocesses to use for data loading
		# label_names=,  # the list of all labels
		load_best_model_at_end=True,  # load the best model found during training at the end of the training
		metric_for_best_model='macro_f1',  # metric to use to compare two different models
		greater_is_better=True,  # whether the `metric_for_best_model` should be maximized or not
		gradient_checkpointing=False,  # use gradient checkpointing to save memory at the expense of slower backward pass
		auto_find_batch_size=False,  # automatically find the best batch size
	)

	print(f"Trainer initializing......")
	trainer = Trainer(
		model=model,  # the instantiated ???? Transformers model to be trained
		args=training_args,  # training arguments, defined above
		train_dataset=train_dataset,  # training dataset
		eval_dataset=dev_dataset,  # evaluation dataset
		tokenizer=tokenizer,  # tokenizer
		data_collator=data_collator,  # data collator
		compute_metrics=compute_metrics,  # the callback that computes metrics of interest
		)

	print(f"Start training......")
	trainer.train()

	print(f"Saving best checkpoint to {os.path.join(output_dir, 'best_ckpt/')}")
	model.save_pretrained(os.path.join(output_dir, 'best_ckpt/'))
	print(f"Done.")


def evaluate(model_name_or_path, eval_fn, typeset_fn, ckpt_dir, max_len, per_device_eval_batch_size,
			 encode_mode: str, num_clsf_layers: int, labels_key: str = 'entgraph_labels', reload_data: bool = False,
			 force_encode: bool = False):

	with open(typeset_fn, 'r', encoding='utf8') as tfp:
		types_list = json.load(tfp)
		assert isinstance(types_list, list)
		num_labels = len(types_list)

	tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, do_lower_case=True)
	config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
	config.update({'encode_mode': encode_mode})
	config.update({'num_clsf_layers': num_clsf_layers})
	model = BertForFiger.from_pretrained(ckpt_dir, config=config)

	eval_dataset = FigerDataset(eval_fn, typeset_fn, tokenizer, len(types_list), 0.0, max_len, labels_key,
								reload_data=reload_data, force_encode=force_encode)

	training_args = TrainingArguments(
		output_dir=ckpt_dir,  # output directory
		per_device_eval_batch_size=per_device_eval_batch_size,  # batch size for evaluation
		dataloader_num_workers=4,  # number of subprocesses to use for data loading
		eval_accumulation_steps=10,
		# label_names=types_list,  # the list of all labels
	)

	trainer = Trainer(
		model=model,  # the instantiated ???? Transformers model to be trained
		args=training_args,  # training arguments, defined above
		eval_dataset=eval_dataset,  # evaluation dataset
		tokenizer=tokenizer,  # tokenizer
		data_collator=data_collator,  # data collator
		compute_metrics=compute_metrics,  # the callback that computes metrics of interest
	)

	print(f"Start evaluating......")
	trainer.evaluate()


# def predict_wtrainer(model_name_or_path, test_fn, test_out_fn, typeset_fn, ckpt_dir, max_len, per_device_eval_batch_size,
# 					 encode_mode: str, num_clsf_layers: int, threshold: float, labels_key: str = 'entgraph_labels',
# 					 reload_data: bool = False, is_inference: bool = False, predict_half: str = None, debug: bool = False):
#
# 	with open(typeset_fn, 'r', encoding='utf8') as tfp:
# 		types_list = json.load(tfp)
# 		assert isinstance(types_list, list)
# 		num_labels = len(types_list)
#
# 	tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, do_lower_case=True)
# 	config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
# 	config.update({'encode_mode': encode_mode})
# 	config.update({'num_clsf_layers': num_clsf_layers})
# 	model = BertForFiger.from_pretrained(ckpt_dir, config=config)
# 	model.to('cuda')
# 	model = torch.nn.DataParallel(model)
# 	model.eval()
#
# 	if predict_half is None:
# 		test_dataset = FigerDataset(test_fn, typeset_fn, tokenizer, len(types_list), 0.0, max_len, labels_key,
# 								    reload_data=reload_data, half_data=None)
# 	elif predict_half == 'first':
# 		test_dataset = FigerDataset(test_fn, typeset_fn, tokenizer, len(types_list), 0.0, max_len, labels_key,
# 									reload_data=reload_data, half_data='first')
# 	elif predict_half == 'second':
# 		test_dataset = FigerDataset(test_fn, typeset_fn, tokenizer, len(types_list), 0.0, max_len, labels_key,
# 									reload_data=reload_data, half_data='second')
# 	else:
# 		raise ValueError(f"predict_half must be one of None, 'first', 'second', but got {predict_half}")
#
# 	training_args = TrainingArguments(
# 		output_dir=ckpt_dir,  # output directory
# 		per_device_eval_batch_size=per_device_eval_batch_size,  # batch size for evaluation
# 		dataloader_num_workers=4,
# 		eval_accumulation_steps=20,
# 		# label_names=types_list,  # the list of all labels
# 	)
#
# 	trainer = Trainer(
# 		model=model,  # the instantiated ???? Transformers model to be trained
# 		args=training_args,  # training arguments, defined above
# 		eval_dataset=test_dataset,  # evaluation dataset
# 		tokenizer=tokenizer,  # tokenizer
# 		data_collator=data_collator,  # data collator
# 		compute_metrics=None if is_inference else compute_metrics,  # the callback that computes metrics of interest
# 	)
#
# 	print(f"Start predicting......")
# 	predictions = trainer.predict(test_dataset)
# 	if debug:
# 		print(predictions)
# 		print(predictions.predictions.shape)
#
# 	out_fp = open(test_out_fn, 'w', encoding='utf8')
# 	for (pred, entry) in zip(predictions.predictions, test_dataset):
# 		pred = np.where(pred > threshold, 1, 0)
# 		assert len(pred) == num_labels
# 		out_item = {
# 			'id': entry['id'].decode('ascii'),
# 			'sentid': entry['sentid'].decode('ascii'),
# 			'fileid': entry['fileid'].decode('ascii'),
# 			'entity_name': entry['entity_name'].decode('ascii'),
# 			'type_preds': [types_list[i] for i in range(num_labels) if pred[i] == 1],
# 		}
# 		if 'labels' in entry:
# 			assert all(x in [0, 1] for x in entry['labels'])
# 			assert len(entry['labels']) == num_labels
# 			out_item['labels'] = [types_list[i] for i in range(num_labels) if entry['labels'][i] == 1]
# 		out_line = json.dumps(out_item, ensure_ascii=False)
# 		out_fp.write(out_line + '\n')
#
# 	out_fp.close()


def predict(model_name_or_path, test_fn, test_out_fn, typeset_fn, ckpt_dir, max_len, per_device_eval_batch_size,
			encode_mode: str, num_clsf_layers: int, threshold: float, labels_key: str = 'entgraph_labels', reload_data: bool = False,
			is_inference: bool = True, predict_half: str = None, spanend_inclusive: bool = False, debug: bool = False,
			no_cuda: bool = False, force_encode: bool = False):

	with open(typeset_fn, 'r', encoding='utf8') as tfp:
		types_list = json.load(tfp)
		assert isinstance(types_list, list)
		num_labels = len(types_list)

	tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, do_lower_case=True)
	config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
	config.update({'encode_mode': encode_mode})
	config.update({'num_clsf_layers': num_clsf_layers})
	model = BertForFiger.from_pretrained(ckpt_dir, config=config)
	model.eval()
	use_cuda_flag = torch.cuda.is_available() and not no_cuda

	if use_cuda_flag:
		model.to('cuda')
		model = torch.nn.DataParallel(model)
	else:
		pass
	n_gpus = torch.cuda.device_count()
	bsz = per_device_eval_batch_size * max(1, n_gpus)

	out_fp = open(test_out_fn, 'w', encoding='utf8')

	assert predict_half is None
	test_dataloader = get_dataloader(test_fn, typeset_fn, num_labels, tokenizer, bsz, max_len, shuffle=False,
									smooth_labels=0.0, labels_key=labels_key, reload_data=reload_data, debug=debug,
									 cache_mode='hdf5', with_labels=not is_inference, num_workers=4,
									 spanend_inclusive=spanend_inclusive, force_encode=force_encode)

	print(f"Start predicting...... Outputting to {test_out_fn}")
	for bidx, batch in enumerate(test_dataloader):
		if debug or bidx % 100 == 0:
			print(f"batch {bidx} / {len(test_dataloader)}; {bidx / len(test_dataloader) * 100:.2f}%")
		if use_cuda_flag:
			batch = {k: v if isinstance(v, list) else v.to('cuda') for k, v in batch.items()}
		with torch.no_grad():
			outputs = model(**batch)
		if len(outputs[0].shape) == 0:
			preds = outputs[1]
		else:
			preds = outputs[0]
		assert preds.shape[1] == num_labels
		preds = torch.softmax(preds, dim=1)
		preds = preds.cpu().numpy()
		# print(preds)
		# pred_labels = np.where(preds > threshold, 1, 0)
		# print(preds)
		# print(preds.sum(axis=1))
		# print(pred_labels)
		# print(pred_labels.shape)
		# print(batch)
		# assert pred_labels.shape[0] == len(batch['id']), f"pred labels shape: {pred_labels.shape} vs ids length: {len(batch['id'])}"
		assert preds.shape[0] == len(
			batch['id']), f"pred labels shape: {preds.shape} vs ids length: {len(batch['id'])}"

		for entidx, scrs in enumerate(preds):
			out_item = {
				'id': batch['id'][entidx].decode('ascii'),
				'sentid': batch['sentid'][entidx].decode('ascii'),
				'fileid': batch['fileid'][entidx].decode('ascii'),
				'entity_name': batch['entity_name'][entidx].decode('ascii'),
				'type_preds': [(types_list[i], float(scrs[i])) for i in range(num_labels) if scrs[i] > threshold],
			}
			# print(f"out preds: {out_item['type_preds']}")
			if 'labels' in batch:
				assert all(x in [0, 1] for x in batch['labels'][entidx])
				assert len(batch['labels'][entidx]) == num_labels
				out_item['labels'] = [types_list[i] for i in range(num_labels) if batch['labels'][entidx][i] == 1]
			out_line = json.dumps(out_item, ensure_ascii=False)
			out_fp.write(out_line + '\n')

	out_fp.close()
	print("Done predicting......")


def build_cache(model_name_or_path: str, train_fn: str, dev_fn: str, test_fn: str, typeset_fn: str, num_labels: int,
				max_len: int, labels_key: str, label_smoothing_factor: float = 0.0, reload_data: bool = False,
				spanend_inclusive: bool = False, force_encode: bool = False):
	tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, do_lower_case=True)

	print(f"Building training set entries......")
	train_dataset = FigerDataset(train_fn, typeset_fn, tokenizer, num_labels, label_smoothing_factor, max_len,
								 labels_key, reload_data=reload_data, spanend_inclusive=spanend_inclusive,
								 force_encode=force_encode)

	print(f"Building dev set entries......")
	dev_dataset = FigerDataset(dev_fn, typeset_fn, tokenizer, num_labels, label_smoothing_factor, max_len, labels_key,
							   reload_data=reload_data, spanend_inclusive=spanend_inclusive, force_encode=force_encode)

	print(f"Building test set entries......")
	test_dataset = FigerDataset(test_fn, typeset_fn, tokenizer, num_labels, label_smoothing_factor, max_len, labels_key,
								reload_data=reload_data, spanend_inclusive=spanend_inclusive, force_encode=force_encode)

	print(f"Done!")


def build_single_cache(model_name_or_path: str, fn: str, typeset_fn: str, num_labels: int,
					   max_len: int, labels_key: str, label_smoothing_factor: float = 0.0,
					   reload_data: bool = False, span_inclusive: bool = False, force_encode: bool = False):
	tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, do_lower_case=True)

	print(f"Building {fn} entries......")
	dataset = FigerDataset(fn, typeset_fn, tokenizer, num_labels, label_smoothing_factor, max_len, labels_key,
						   reload_data=reload_data, with_labels=False, spanend_inclusive=span_inclusive,
						   force_encode=force_encode)

	print(f"Done!")


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_name_or_path', type=str, default='bert-base-uncased')
	parser.add_argument('--encode_mode', type=str, default='entity')
	parser.add_argument('--num_clsf_layers', type=int, default=1)
	parser.add_argument('--data_dir', type=str, default='../json_data')
	parser.add_argument('--train_fn', type=str, default='train_wegtypes.json')
	parser.add_argument('--dev_fn', type=str, default='dev_wegtypes.json')
	parser.add_argument('--test_fn', type=str, default='test_wegtypes.json')
	parser.add_argument('--typeset_fn', type=str, default='../raw_data/figer2entgraph_typemap.json.set')
	parser.add_argument('--output_dir', type=str, default='../model_ckpts/')
	parser.add_argument('--max_len', type=int, default=128)
	parser.add_argument('--per_device_train_batch_size', type=int, default=32)
	parser.add_argument('--per_device_eval_batch_size', type=int, default=64)
	parser.add_argument('--num_train_epochs', type=int, default=5)
	parser.add_argument('--lr', type=float, default=5e-5)
	parser.add_argument('--weight_decay', type=float, default=0.01)
	parser.add_argument('--warmup_ratio', type=float, default=0.01)
	parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
	parser.add_argument('--logging_steps', type=int, default=5000)
	parser.add_argument('--save_steps', type=int, default=5000)
	parser.add_argument('--seed', type=int, default=42)
	parser.add_argument('--use_fp16', action='store_true')
	# parser.add_argument('--fp16_opt_level', type=str, default='O1')
	# parser.add_argument('--fp16_fulleval', action='store_true')
	parser.add_argument('--metric_best_model', type=str, default='macro_f1')
	parser.add_argument('--label_smoothing_factor', type=float, default=0.0)
	parser.add_argument('--labels_key', type=str, default='entgraph_labels')

	parser.add_argument('--reload_data', action='store_true')
	parser.add_argument('--is_inference', action='store_true')
	parser.add_argument('--do_train', action='store_true')
	parser.add_argument('--do_dev', action='store_true')
	parser.add_argument('--do_test', action='store_true')
	parser.add_argument('--do_predict', action='store_true')
	parser.add_argument('--do_cache', action='store_true')
	parser.add_argument('--do_single_cache', action='store_true')

	parser.add_argument('--predict_fn', type=str, default=None)
	parser.add_argument('--predict_threshold', type=float, default=None)
	parser.add_argument('--debug', action='store_true')
	parser.add_argument('--no_cuda', action='store_true')
	parser.add_argument('--predict_half', type=str, default=None)
	parser.add_argument('--spanend_inclusive', action='store_true')
	parser.add_argument('--force_encode', action='store_true')

	args = parser.parse_args()
	print(f"args: {args}")

	if torch.cuda.is_available():
		print(f"CUDA is available!")
	else:
		print(f"CUDA is not available!")
		if args.do_train or args.do_predict:
			# raise Exception("CUDA is not available!")
			print(f"Training or predicting without CUDA!")

	args.train_fn = os.path.join(args.data_dir, args.train_fn)
	args.dev_fn = os.path.join(args.data_dir, args.dev_fn)
	args.test_fn = os.path.join(args.data_dir, args.test_fn)
	assert args.predict_fn is None or args.predict_fn[-5:] == '.json'
	args.predict_fn = os.path.join(args.data_dir, args.predict_fn) if args.predict_fn is not None else None
	args.predict_outfn = args.predict_fn[:-5] + '_preds.json' if args.predict_fn is not None else None

	lm_model_name = args.model_name_or_path.split("/")[-1]
	types_list = get_labelset(args.typeset_fn)

	if args.do_cache:
		build_cache(args.model_name_or_path, args.train_fn, args.dev_fn, args.test_fn, args.typeset_fn,
					len(types_list), args.max_len, args.labels_key, label_smoothing_factor=args.label_smoothing_factor,
					reload_data=args.reload_data, spanend_inclusive=args.spanend_inclusive, force_encode=args.force_encode)

	if args.do_single_cache:
		build_single_cache(args.model_name_or_path, args.predict_fn, args.typeset_fn,
						   len(types_list), args.max_len, args.labels_key,
						   label_smoothing_factor=args.label_smoothing_factor, reload_data=args.reload_data,
						   span_inclusive=args.spanend_inclusive, force_encode=args.force_encode)

	if args.do_train:
		fpxx_str = 'fp16' if args.use_fp16 else 'fp32'
		args.output_dir = os.path.join(args.output_dir, lm_model_name,
									   f"{args.encode_mode}_{args.num_clsf_layers}layers_{args.lr}_{args.label_smoothing_factor}_{fpxx_str}")
		if not os.path.exists(args.output_dir):
			os.makedirs(args.output_dir)
		train(model_name_or_path=args.model_name_or_path, types_list=types_list, encode_mode=args.encode_mode,
			  num_clsf_layers=args.num_clsf_layers,
			  train_fn=args.train_fn, dev_fn=args.dev_fn, typeset_fn=args.typeset_fn, output_dir=args.output_dir,
			  max_len=args.max_len, per_device_train_batch_size=args.per_device_train_batch_size,
			  per_device_eval_batch_size=args.per_device_eval_batch_size, num_train_epochs=args.num_train_epochs,
			  learning_rate=args.lr, weight_decay=args.weight_decay, warmup_ratio=args.warmup_ratio,
			  gradient_accumulation_steps=args.gradient_accumulation_steps, save_steps=args.save_steps,
			  logging_steps=args.logging_steps, seed=args.seed, use_fp16=args.use_fp16, metric_best_model=args.metric_best_model,
			  label_smoothing_factor=args.label_smoothing_factor,
			  labels_key=args.labels_key, reload_data=args.reload_data, debug=args.debug, force_encode=args.force_encode)
	else:
		pass

	if args.do_dev:
		print(f"Doing evaluation on dev set...")
		evaluate(model_name_or_path=args.model_name_or_path, eval_fn=args.dev_fn, typeset_fn=args.typeset_fn,
				 ckpt_dir=args.output_dir, max_len=args.max_len, per_device_eval_batch_size=args.per_device_eval_batch_size,
				 encode_mode=args.encode_mode, num_clsf_layers=args.num_clsf_layers, labels_key=args.labels_key,
				 reload_data=args.reload_data, force_encode=args.force_encode)
		print(f"Done.")
	else:
		pass

	if args.do_test:
		print(f"Doing evaluation on test set...")
		evaluate(model_name_or_path=args.model_name_or_path, eval_fn=args.test_fn, typeset_fn=args.typeset_fn,
				 ckpt_dir=args.output_dir, max_len=args.max_len, per_device_eval_batch_size=args.per_device_eval_batch_size,
				 encode_mode=args.encode_mode, num_clsf_layers=args.num_clsf_layers, labels_key=args.labels_key,
				 reload_data=args.reload_data, force_encode=args.force_encode)
		print(f"Done.")

	if args.do_predict:
		print(f"Doing prediction over {args.predict_fn}; storing prediction outputs in {args.predict_outfn}...")
		predict(model_name_or_path=args.model_name_or_path, test_fn=args.predict_fn, test_out_fn=args.predict_outfn,
				typeset_fn=args.typeset_fn, ckpt_dir=args.output_dir, max_len=args.max_len,
				per_device_eval_batch_size=args.per_device_eval_batch_size, encode_mode=args.encode_mode,
				num_clsf_layers=args.num_clsf_layers,
				threshold=args.predict_threshold, labels_key=args.labels_key, reload_data=args.reload_data,
				is_inference=args.is_inference, predict_half=args.predict_half, spanend_inclusive=args.spanend_inclusive,
				debug=args.debug, no_cuda=args.no_cuda, force_encode=args.force_encode)
		print(f"Done.")
	else:
		pass
