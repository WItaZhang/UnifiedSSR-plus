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



if __name__ == "__main__":
	args = parse_args()

	# Seed
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	# Evaluation
	# args.tasks and args.trained_model_path are set via --tasks and --trained_model_path arguments
	args.phase = 'finetune'

	if not args.save_dir:
		args.save_dir = os.path.dirname(args.trained_model_path)

	result_columns = ['model_name'] + [f'HR@{k}' for k in args.k_list] + [f'NDCG@{k}' for k in args.k_list]
	all_results = pd.DataFrame(columns=result_columns)

	model_dir = os.path.dirname(args.trained_model_path)
	pth_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]

	for pth_file in sorted(pth_files, reverse=True):
		logging.info(f"\nProcessing model: {pth_file}")
		args.trained_model_path = os.path.join(model_dir, pth_file)
		hits, ndcgs = predict_product(args)

		result_row = [pth_file] + [round(hit, 4) for hit in hits.tolist()] + [round(ndcg, 4) for ndcg in ndcgs.tolist()]
		all_results.loc[len(all_results)] = result_row

	task_name = '_'.join(args.tasks)
	result_file = os.path.join(args.save_dir, f'all_results_{task_name}.csv')
	all_results.to_csv(result_file, index=False)
	logging.info(f"\nAll results saved to {result_file}")
