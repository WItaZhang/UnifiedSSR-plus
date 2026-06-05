from utils.parser import parse_args
from utils.logger import create_log_id, logging_config
from utils.optimizer import NoamOpt
from utils.utils import save_model, load_model
from data import UnifiedDataset
from batch import BatchSampler, collate_pretrain
from model import get_model
import os, time, logging, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pretrain import run_epoch


def continue_pretrain(args):
    # log
    log_name = 'log_continue_pretrain'
    log_save_id = create_log_id(args.save_dir, name=log_name)
    logging_config(folder=args.save_dir, name='{}_{:d}'.format(log_name, log_save_id), no_console=False)
    logging.info(args)

    # GPU / CPU
    args.use_cuda = args.use_cuda & torch.cuda.is_available()
    device = torch.device("cuda:{}".format(args.cuda_idx) if args.use_cuda else "cpu")

    # load data
    data = UnifiedDataset(args.phase, args.tasks, args.data_root, logging)

    batch_sampler = BatchSampler(data, args.train_batch_size)
    data_loader = DataLoader(data,
                           batch_sampler=batch_sampler,
                           collate_fn=lambda x: collate_pretrain(x, args))
    batch_num = len(data_loader)

    # construct model and load previous checkpoint
    model = get_model(args)
    model.to(device)
    
    if not os.path.isfile(args.trained_model_path):
        raise ValueError(f"Model checkpoint not found at {args.trained_model_path}")
    
    logging.info(f"Loading previous model checkpoint: {args.trained_model_path}")
    model = load_model(model, args.trained_model_path)
    
    # define optimizer and load previous state
    optimizer = NoamOpt(args.emb_size, args.opt_factor, args.opt_warmup,
                       torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    
    optimizer_path = os.path.join(os.path.dirname(args.trained_model_path), 'optimizer.pth')
    if os.path.isfile(optimizer_path):
        logging.info(f"Loading optimizer state from: {optimizer_path}")
        optimizer_state = torch.load(optimizer_path)
        optimizer.load_state_dict(optimizer_state)
    else:
        logging.warning(f"Optimizer state file not found at {optimizer_path}")
    
    logging.info(optimizer)

    # Continue training from the next epoch
    last_epoch = int(os.path.basename(args.trained_model_path).split('_')[-1].split('.')[0])
    start_epoch_idx = last_epoch + 1
    
    for epoch_idx in range(start_epoch_idx, start_epoch_idx + args.num_epoch):
        # train and save model
        run_epoch(args, model, data_loader, optimizer, epoch_idx, batch_num, device)
        
        # save optimizer state
        if (epoch_idx % args.save_every) == 0:
            optimizer_save_path = os.path.join(args.save_dir, 'optimizer.pth')
            save_model(model, args.save_dir, epoch_idx)
            # torch.save(optimizer.state_dict(), optimizer_save_path)


if __name__ == '__main__':
    args = parse_args()

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Continue Pretrain
    args.phase = 'pretrain'
    args.tasks = ['recommendation', 'search']
    args.trained_model_path = "models/Amazon_Electronics/pretrain_recommendation_search/20250218_170242/model_160.pth"
    
    # Use the same save_dir as the original training
    args.save_dir = os.path.dirname(args.trained_model_path)
    args.num_epoch=40
    continue_pretrain(args)