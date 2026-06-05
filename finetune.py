from utils.parser import parse_args
from utils.logger import create_log_id, logging_config
from utils.optimizer import NoamOpt
from utils.utils import save_model, load_model, early_stopping
from metrics import evaluate_product
from data import UnifiedDataset
from batch import BatchSampler, collate_train, collate_val
from model import get_model
from pretrain import run_epoch
import numpy as np
import pandas as pd
import os, time, logging, random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.parser import parse_args
from utils.logger import create_log_id, logging_config
from utils.utils import load_model
from metrics import evaluate_product
from data import UnifiedDataset
from batch import collate_test
from model import get_model
import numpy as np
import pandas as pd
import os, logging, random
import torch
from torch.utils.data import DataLoader

def predict_product(args):
	# log
	log_name = f'log_test_{args.tasks[0]}'
	log_save_id = create_log_id(args.save_dir, name=log_name)
	logging_config(folder=args.save_dir, name='{}_{:d}'.format(log_name, log_save_id), no_console=False)
	logging.info(args)

	# GPU / CPU
	args.use_cuda = args.use_cuda & torch.cuda.is_available()
	device = torch.device("cuda:{}".format(args.cuda_idx) if args.use_cuda else "cpu")

	# load data
	data = UnifiedDataset(args.phase, args.tasks, args.data_root, logging)

	data_loader = DataLoader(data,
	                         shuffle=False,
	                         batch_size=args.test_batch_size,
	                         collate_fn=lambda x: collate_test(x, args))

	# load model
	model = get_model(args)
	model = load_model(model, args.trained_model_path).to(device)

	# evaluate
	hits, ndcgs = evaluate_product(model, data_loader, len(data), args, device)
	for k_idx, topk in enumerate(args.k_list):
		logging.info(
			'Evaluation (K={}): HR {:.4f} NDCG {:.4f}'.format(topk, hits[k_idx], ndcgs[k_idx]))

	# initialize metrics
	result_save_file = os.path.join(args.save_dir, 'test_results.csv')
	init_metrics = pd.DataFrame(['HR@{}'.format(k) for k in args.k_list] +
	                            ['NDCG@{}'.format(k) for k in args.k_list]).transpose()
	init_metrics.to_csv(result_save_file, mode='a', header=False, sep='\t', index=False)
	metrics = pd.DataFrame(hits.tolist() + ndcgs.tolist()).transpose()
	metrics.to_csv(result_save_file, mode='a', header=False, sep='\t', index=False)
	return hits, ndcgs




def train(args):
	# log
	log_name = 'log_train'
	log_save_id = create_log_id(args.save_dir, name=log_name)
	logging_config(folder=args.save_dir, name='{}_{:d}'.format(log_name, log_save_id), no_console=False)
	logging.info(args)

	# GPU / CPU
	args.use_cuda = args.use_cuda & torch.cuda.is_available()
	device = torch.device("cuda:{}".format(args.cuda_idx) if args.use_cuda else "cpu")

	# load data
	train_data = UnifiedDataset(args.phase, args.tasks, args.data_root, logging)

	batch_sampler = BatchSampler(train_data, args.train_batch_size)
	data_loader = DataLoader(train_data,
	                         batch_sampler=batch_sampler,
	                         collate_fn=lambda x: collate_train(x, args))
	batch_num = len(data_loader)

	# construct model
	model = get_model(args)
	model.to(device)
	logging.info(model)

	if os.path.isfile(args.trained_model_path):
		logging.info("Loading pre-trained model: {}".format(args.trained_model_path))
		model = load_model(model, args.trained_model_path)
	else:
		logging.info('Parameters initializing ...')
		for p in model.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_((p))
		model.seq_partition.reset_offset()

	# define optimizer
	optimizer = NoamOpt(args.emb_size, args.opt_factor, args.opt_warmup,
	                    torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
	logging.info(optimizer)

	# train
	init_metrics = pd.DataFrame(['Epoch_idx'] + ['HR@{}'.format(k) for k in args.k_list] +
	                            ['NDCG@{}'.format(k) for k in args.k_list]).transpose()
	init_metrics.to_csv(os.path.join(args.save_dir, 'train_results.csv'), mode='a', header=False,
	                    sep='\t', index=False)
	cur_best_scores = [0., 0.]
	stopping_count = 0
	should_stop = False
	best_epoch_idx = -1
	best_result = -np.inf
	best_results = []

	assert len(args.tasks) == 1, 'Phase specify a single downstream task to train & test.'
	start_epoch_idx = args.start_epoch_idx or 1
	for epoch_idx in range(start_epoch_idx, args.num_epoch + start_epoch_idx):
		# train and save model
		run_epoch(args, model, data_loader, optimizer, epoch_idx, batch_num, device)

		# evaluate
		if (epoch_idx % args.evaluate_every) == 0:
			time3 = time.time()
			val_data = UnifiedDataset(args.phase, args.tasks, args.data_root, logging)
			val_data_loader = DataLoader(val_data,
			                             shuffle=False,
			                             batch_size=args.test_batch_size,
			                             collate_fn=lambda x: collate_val(x, args))

			hits, ndcgs = evaluate_product(model, val_data_loader, len(val_data), args, device)
			for k_idx, topk in enumerate(args.k_list):
				logging.info('Evaluation (K={}): Epoch {:04d} | Total Time {:.1f}s | HR {:.4f} NDCG {:.4f}'.format(
					topk, epoch_idx, time.time() - time3, hits[k_idx], ndcgs[k_idx]))

			# cur_best_scores, stopping_count, should_stop = early_stopping([hits[-1], ndcgs[-1]], cur_best_scores, stopping_count, 3, logging)

			# save the best result
			if ndcgs[0] > best_result:
				best_result = ndcgs[0]
				best_results = hits.tolist() + ndcgs.tolist()
				save_model(model, args.save_dir, epoch_idx, best_epoch_idx)
				best_epoch_idx = epoch_idx

			metrics = pd.DataFrame([epoch_idx] + hits.tolist() + ndcgs.tolist()).transpose()
			metrics.to_csv(os.path.join(args.save_dir, 'train_results.csv'), mode='a', header=False, sep='\t',
			               index=False)

		if should_stop == True:
			break

	best_metrics = pd.DataFrame([best_epoch_idx] + best_results).transpose()
	best_metrics.to_csv(os.path.join(args.save_dir, 'train_results.csv'), mode='a', header=False, sep='\t', index=False)


if __name__ == '__main__':
	args = parse_args()

	# Seed
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	# Fine-tune and evaluate
	args.phase = 'finetune'
	# args.tasks is set via --tasks argument (default: ['search'])

	if not args.save_dir:
		args.save_dir = os.path.join('models', args.data_name, f'{"_".join([args.phase] + args.tasks)}', time.strftime("%Y%m%d_%H%M%S"))
	train(args)
