import argparse
import pandas as pd
import os
import sys
sys.path.append(os.getcwd())


def parse_args():
    parser = argparse.ArgumentParser()

    # basic
    parser.add_argument('--seed', type=int, default=888)
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--cuda_idx', type=int, default=2)

    # pretrain and train
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--train_num_neg', type=int, default=4)
    parser.add_argument('--start_epoch_idx', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--opt_factor', type=float, default=1)
    parser.add_argument('--opt_warmup', type=int, default=4000)
    parser.add_argument('--print_every', type=int, default=1,
                        help='Iteration interval of printing loss.')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Iteration interval of saving model.')
    parser.add_argument('--evaluate_every', type=int, default=20,
                        help='Epoch interval of evaluation.')

    # validation and test
    parser.add_argument('--test_batch_size', type=int, default=20)
    parser.add_argument('--test_num_neg', type=int, default=99)
    parser.add_argument('--test_neg', type=bool, default=False)
    parser.add_argument('--k_list', type=list, default=[5, 10,50,100])

    # model
    parser.add_argument('--corr_factor', type=float, default=0.1)
    parser.add_argument('--num_head', type=int, default=4,
                        choices=[1, 2, 4, 8])
    parser.add_argument('--enc_num_layer', type=int, default=1,
                        choices=[1, 2, 3, 4, 5, 6])
    parser.add_argument('--sub_seq_num', type=int, default=1,
                        choices=[1, 2, 3, 4,5,6,7,8])
    parser.add_argument('--emb_size', type=int, default=112,
                        choices=[16, 32, 48, 64, 80,96,112,128,144,160])
    parser.add_argument('--hid_size', type=int, default=None)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--trained_model_path', type=str, default='',
                        help='Path to a pretrained model checkpoint (.pth) to load before fine-tuning.')
    parser.add_argument('--save_dir', type=str, default='',
                        help='Directory to save model checkpoints and logs.')
    parser.add_argument('--tasks', type=str, nargs='+', default=['search'],
                        choices=['search', 'recommendation'],
                        help='Task(s) to train/evaluate on.')

    # data
    parser.add_argument('--data_name', type=str, default='Amazon_Electronics_WholeQuery', choices=['JDsearch','JDsearch2', 'Amazon_Clothing', 'Amazon_Electronics','Amazon_Electronics7','KuaiSAR_v2_5','KuaiSAR_v2_3','Amazon_Electronics_WholeQuery','Amazon_Electronics_Random'])
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--padding_value', type=int, default=0)
    parser.add_argument('--query_max_len', type=int, default=50)

    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='Number of gradient accumulation steps before a backward/update pass.')

    args = parser.parse_args()

    args.data_root = os.path.join(args.data_root, args.data_name)
    data_meta_path = os.path.join(args.data_root, 'meta.csv')
    user_num, product_num, term_num,query_num = pd.read_csv(data_meta_path, sep='\t').values.squeeze()
    args.user_vocab = user_num + 1
    args.product_vocab = product_num + 1
    args.term_vocab = term_num + 1
    args.query_vocab = query_num + 1
    args.bos_id = args.term_vocab + 1 # Begin-of-Sentence
    args.eos_id = args.term_vocab # End-of-Sentence

    return args


if __name__ == "__main__":
    # for test only
    pass