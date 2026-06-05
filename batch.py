import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Sampler
from torch.nn.utils.rnn import pad_sequence


def subsequent_mask(size):
    """
    生成用于注意力机制的下三角掩码。

    此函数旨在创建一个掩码矩阵，用于在不应关注后续信息的场景中，例如自回归模型的训练过程中。
    掩码允许每个位置关注自身及之前的所有位置，但不能关注后续位置。

    参数:
    - size: 注意力图的长度和宽度大小，表示最大序列长度。

    返回:
    - 一个形状为 (1, size, size) 的布尔张量，其中下三角部分（包括对角线）为 True，
      其余部分为 False。此张量用于在注意力机制中屏蔽后续位置。
    """
    # 创建一个形状为 (1, size, size) 的全1方阵
    # 使用 tril 函数保留主对角线及其以下的元素，其余设置为 0
    # 将矩阵转换为布尔类型，其中 1 为 True，0 为 False
    return torch.tril(torch.ones((1, size, size)), diagonal=0).bool()


def neg_sampling(batch_data, num_neg, vocab):
    """
    执行负采样操作。

    对于给定的一批数据，为每个正样本生成指定数量的负样本。负样本是从给定的词汇表中随机选择的，
    并确保负样本不在原始数据中。

    参数:
    batch_data: 一批数据，可以是单个元素或元素的列表。
    num_neg: 需要为每个正样本生成的负样本数量。
    vocab: 词汇表大小，负样本将从[1, vocab)的范围内生成。

    返回:
    batch_neg: 采样后的包含负样本的列表。
    """
    # 初始化存储负样本的列表
    batch_neg = []
    # 遍历输入的数据批次
    for data in batch_data:
        # 如果当前数据是列表类型，递归调用负采样函数
        if isinstance(data, list):
            batch_neg.append(neg_sampling(data, num_neg, vocab))
        else:
            # 初始化存储当前数据负样本的列表
            sampled_negs = []
            # 生成指定数量的负样本
            for _ in range(num_neg):
                # 随机生成一个负样本
                neg = np.random.randint(1, vocab)
                # 确保负样本既不与正样本重复，也不与其他负样本重复
                while neg == data or neg in sampled_negs:
                    neg = np.random.randint(1, vocab)
                # 将验证后的负样本添加到列表中
                sampled_negs.append(neg)
            # 将当前数据的所有负样本添加到最终的负样本列表中
            batch_neg.append(sampled_negs)
    # 返回包含所有负样本的列表
    return batch_neg


def collate_pretrain(batch_data, args):
    """
    处理预训练数据集中的一个批次数据。

    根据不同的任务类型（推荐或搜索），对用户ID、产品ID序列进行不同的处理和负采样，以准备模型训练所需的数据。

    参数:
    - batch_data: 一个批次的数据，包含多个用户的交互记录。
    - args: 包含训练参数和配置的对象。

    返回:
    - 任务类型和一个包含相应数据张量的字典。
    """
    # 获取负样本数量、产品词汇表和填充值
    num_neg = args.train_num_neg
    p_vocab = args.product_vocab
    padding_value = args.padding_value

    # 提取并转换用户ID到张量
    uids = [int(data['uid']) for data in batch_data]
    uids = torch.tensor(uids).long()

    # 根据任务标志处理数据
    if batch_data[0]['flag'] == 'recommendation':
        # 处理推荐任务的数据
        pids_in = [torch.tensor(data['pid_list'][:-1]).long() for data in batch_data]
        pids_in = pad_sequence(pids_in, batch_first=True, padding_value=padding_value)
        pids_out = [data['pid_list'][1:] for data in batch_data]
        pids_out_neg = neg_sampling(pids_out, num_neg, p_vocab)
        pids_out = pad_sequence([torch.tensor(pid_out).long() for pid_out in pids_out], batch_first=True,
                                padding_value=padding_value)
        pids_out_neg = pad_sequence([torch.tensor(pid_neg).long() for pid_neg in pids_out_neg], batch_first=True,
                                    padding_value=padding_value)
        pids_mask = (pids_in != padding_value).unsqueeze(-2)
        pid_last_idx = torch.tensor([data['length'] - 2 for data in batch_data])
        return 'recommendation', {'uid': uids,
                                  'pids_in': pids_in,
                                  'pids_mask': pids_mask,
                                  'pids_tgt': pids_out,
                                  'pids_neg': pids_out_neg,
                                  'pid_last_idx': pid_last_idx}


    else:  # 'search'
        # 处理搜索任务的数据
        pids_in = [torch.tensor(data['pid_list'][:-1]).long() for data in batch_data]
        pids_in = pad_sequence(pids_in, batch_first=True, padding_value=padding_value)
        # pids_in = F.pad(pids_in, (1, 0), value=padding_value)
        pids_out = [data['pid_list'][1:] for data in batch_data]
        pids_out_neg = neg_sampling(pids_out, num_neg, p_vocab)
        pids_out = pad_sequence([torch.tensor(pid_out).long() for pid_out in pids_out], batch_first=True,
                                padding_value=padding_value)
        pids_out_neg = pad_sequence([torch.tensor(pid_neg).long() for pid_neg in pids_out_neg], batch_first=True,
                                    padding_value=padding_value)
        pids_mask = (pids_in != padding_value).unsqueeze(-2)
        pid_last_idx = torch.tensor([data['length'] - 2 for data in batch_data])

        # 处理查询数据
        qrys_in = [data['qry_ids'] for data in batch_data]  # Assuming each query now has a unique ID
        qrys_in_seq_max_len = max(len(qrys) for qrys in qrys_in)
        qrys_in = pad_sequence([torch.tensor(qrys).long() for qrys in qrys_in], 
                             batch_first=True, 
                             padding_value=padding_value)
        qrys_in_mask = (qrys_in != padding_value).unsqueeze(-2)
        # qrys_in [BS, Seq Max Len, Qry Len]

        return 'search', {'uid': uids,
                          'pids_in': pids_in,
                          'pids_mask': pids_mask,
                          'pids_tgt': pids_out,
                          'pids_neg': pids_out_neg,
                          'qrys_in': qrys_in,
                          'qrys_in_mask': qrys_in_mask,
                          'pid_last_idx': pid_last_idx}



def collate_train(batch_data, args):
    """
    处理训练数据批次，根据不同的任务类型（推荐或搜索）对数据进行不同方式的处理和封装。

    参数:
    - batch_data: 一个列表，包含一批数据样本，每个样本是一个字典。
    - args: 包含训练配置和参数的对象，如训练使用的负样本数量和产品词汇表。

    返回:
    - 一个根据任务类型处理后的数据字典，包含用户ID、产品ID序列、掩码等信息。
    """
    # 获取负样本数量、产品词汇表和填充值
    num_neg = args.train_num_neg
    p_vocab = args.product_vocab
    padding_value = args.padding_value

    # 提取并转换用户ID到张量
    uids = [int(data['uid']) for data in batch_data]
    uids = torch.tensor(uids).long()

    # 根据数据批次中的任务标记，决定处理数据的方式
    if batch_data[0]['flag'] == 'recommendation':
        # 处理推荐任务的数据
        pids_in = [torch.tensor(data['pid_list'][:-3]).long() for data in batch_data]
        pids_in = pad_sequence(pids_in, batch_first=True, padding_value=padding_value)
        pids_out = [data['pid_list'][1:-2] for data in batch_data]
        pids_out_neg = neg_sampling(pids_out, num_neg, p_vocab)
        pids_out = pad_sequence([torch.tensor(pid_out).long() for pid_out in pids_out], batch_first=True,
                                padding_value=padding_value)
        pids_out_neg = pad_sequence([torch.tensor(pid_neg).long() for pid_neg in pids_out_neg], batch_first=True,
                                    padding_value=padding_value)
        pids_mask = (pids_in != padding_value).unsqueeze(-2)
        pid_last_idx = torch.tensor([data['length'] - 4 for data in batch_data])
        return 'recommendation', {'uid': uids,
                                  'pids_in': pids_in,
                                  'pids_mask': pids_mask,
                                  'pids_tgt': pids_out,
                                  'pids_neg': pids_out_neg,
                                  'pid_last_idx': pid_last_idx}

    else:  # 'search'
        # 处理搜索任务的数据
        pids_in = [torch.tensor(data['pid_list'][:-3]).long() for data in batch_data]
        pids_in = pad_sequence(pids_in, batch_first=True, padding_value=padding_value)
        # pids_in = F.pad(pids_in, (1, 0), value=padding_value)
        pids_out = [data['pid_list'][1:-2] for data in batch_data]
        pids_out_neg = neg_sampling(pids_out, num_neg, p_vocab)
        pids_out = pad_sequence([torch.tensor(pid_out).long() for pid_out in pids_out], batch_first=True,
                                padding_value=padding_value)
        pids_out_neg = pad_sequence([torch.tensor(pid_neg).long() for pid_neg in pids_out_neg], batch_first=True,
                                    padding_value=padding_value)
        pids_mask = (pids_in != padding_value).unsqueeze(-2)
        pid_last_idx = torch.tensor([data['length'] - 4 for data in batch_data])

        # 处理查询数据
        qrys_in = [data['qry_ids'][:-2] for data in batch_data]  # Assuming each query now has a unique ID
        qrys_in_seq_max_len = max(len(qrys) for qrys in qrys_in)
        qrys_in = pad_sequence([torch.tensor(qrys).long() for qrys in qrys_in], 
                             batch_first=True, 
                             padding_value=padding_value)
        qrys_in_mask = (qrys_in != padding_value).unsqueeze(-2)

        return 'search', {'uid': uids,
                          'pids_in': pids_in,
                          'pids_mask': pids_mask,
                          'pids_tgt': pids_out,
                          'pids_neg': pids_out_neg,
                          'qrys_in': qrys_in,
                          'qrys_in_mask': qrys_in_mask,
                          'pid_last_idx': pid_last_idx}



def collate_val(batch_data, args):
    """
    处理验证集数据，根据不同的任务类型（推荐或搜索）对数据进行不同的处理。

    参数:
    - batch_data: 一个列表，包含一批数据，每条数据是一个字典，包含用户ID、产品ID列表、查询词列表等信息。
    - args: 包含一些配置参数的对象，如测试时的负样本数量、产品词汇表和填充值等。

    返回:
    - 一个元组，包含任务类型和一个字典，字典中包含处理后的数据。
    """
    # 获取负样本数量、产品词汇表和填充值
    num_neg = args.test_num_neg
    p_vocab = args.product_vocab
    padding_value = args.padding_value

    # 提取用户ID并转换为张量
    uids = [int(data['uid']) for data in batch_data]
    uids = torch.tensor(uids).long()

    # 根据数据类型（推荐或搜索）处理数据
    if batch_data[0]['flag'] == 'recommendation':
        # 处理推荐任务的数据
        pids_in = [torch.tensor(data['pid_list'][:-2]).long() for data in batch_data]
        pids_in = pad_sequence(pids_in, batch_first=True, padding_value=padding_value)
        pids_mask = (pids_in != padding_value).unsqueeze(-2)
        pid_last_idx = torch.tensor([data['length'] - 3 for data in batch_data])

        pid_tgt = [data['pid_list'][-2] for data in batch_data]
        pid_tgt_neg = neg_sampling(pid_tgt, num_neg, p_vocab)
        pid_tgt = torch.tensor(pid_tgt).long()
        pid_tgt_neg = torch.tensor(pid_tgt_neg).long()
        pid_pred = torch.cat((pid_tgt.unsqueeze(-1), pid_tgt_neg), -1)

        return 'recommendation', {'uid': uids,
                                  'pids_in': pids_in,
                                  'pids_mask': pids_mask,
                                  'pid_pred': pid_pred,
                                  'pid_last_idx': pid_last_idx}

    else:  # 处理搜索任务的数据
        pids_in = [torch.tensor(data['pid_list'][:-2]).long() for data in batch_data]
        pids_in = pad_sequence(pids_in, batch_first=True, padding_value=padding_value)
        # pids_in = F.pad(pids_in, (1, 0), value=padding_value)
        pid_tgt = [data['pid_list'][-2] for data in batch_data]
        pid_tgt_neg = neg_sampling(pid_tgt, num_neg, p_vocab)
        pid_tgt = torch.tensor(pid_tgt).long()
        pid_tgt_neg = torch.tensor(pid_tgt_neg).long()
        pid_pred = torch.cat((pid_tgt.unsqueeze(-1), pid_tgt_neg), -1)
        pids_mask = (pids_in != padding_value).unsqueeze(-2)
        pid_last_idx = torch.tensor([data['length'] - 3 for data in batch_data])

        qrys_in = [data['qry_ids'][:-1] for data in batch_data]  # Assuming each query now has a unique ID
        qrys_in_seq_max_len = max(len(qrys) for qrys in qrys_in)
        qrys_in = pad_sequence([torch.tensor(qrys).long() for qrys in qrys_in], 
                             batch_first=True, 
                             padding_value=padding_value)
        qrys_in_mask = (qrys_in != padding_value).unsqueeze(-2)

        return 'search', {'uid': uids,
                          'pids_in': pids_in,
                          'pids_mask': pids_mask,
                          'pid_pred': pid_pred,
                          'pid_last_idx': pid_last_idx,
                          'qrys_in': qrys_in,
                          'qrys_in_mask': qrys_in_mask}


def collate_test(batch_data, args):
    """
    处理测试批次数据并根据不同的任务类型进行数据预处理。

    参数:
    - batch_data: 一个列表，包含一批数据样本，每个样本是一个字典。
    - args: 包含测试设置和配置的参数对象。

    返回:
    - task_type: 任务类型，可以是'recommendation'或'search'。
    - batch: 一个字典，包含根据任务类型处理后的批次数据张量。
    """
    # 获取负样本数量、产品词汇表和填充值
    num_neg = args.test_num_neg
    p_vocab = args.product_vocab
    padding_value = args.padding_value

    # 提取用户ID并转换为张量
    uids = [int(data['uid']) for data in batch_data]
    uids = torch.tensor(uids).long()

    # 根据数据标记对数据进行不同的处理
    if batch_data[0]['flag'] == 'recommendation':
        # 处理推荐任务的数据
        pids_in = [torch.tensor(data['pid_list'][:-1]).long() for data in batch_data]
        pids_in = pad_sequence(pids_in, batch_first=True, padding_value=padding_value)
        pids_mask = (pids_in != padding_value).unsqueeze(-2)
        pid_last_idx = torch.tensor([data['length'] - 2 for data in batch_data])

        pid_tgt = [data['pid_list'][-1] for data in batch_data]
        pid_tgt_neg = neg_sampling(pid_tgt, num_neg, p_vocab)
        pid_tgt = torch.tensor(pid_tgt).long()
        pid_tgt_neg = torch.tensor(pid_tgt_neg).long()
        pid_pred = torch.cat((pid_tgt.unsqueeze(-1), pid_tgt_neg), -1)

        return 'recommendation', {'uid': uids,
                                  'pids_in': pids_in,
                                  'pids_mask': pids_mask,
                                  'pid_pred': pid_pred,
                                  'pid_last_idx': pid_last_idx}

    else:  # 处理搜索任务的数据
        pids_in = [torch.tensor(data['pid_list'][:-1]).long() for data in batch_data]
        pids_in = pad_sequence(pids_in, batch_first=True, padding_value=padding_value)
        # pids_in = F.pad(pids_in, (1, 0), value=padding_value)
        pid_tgt = [data['pid_list'][-1] for data in batch_data]
        pid_tgt_neg = neg_sampling(pid_tgt, num_neg, p_vocab)
        pid_tgt = torch.tensor(pid_tgt).long()
        pid_tgt_neg = torch.tensor(pid_tgt_neg).long()
        pid_pred = torch.cat((pid_tgt.unsqueeze(-1), pid_tgt_neg), -1)
        pids_mask = (pids_in != padding_value).unsqueeze(-2)
        pid_last_idx = torch.tensor([data['length'] - 2 for data in batch_data])

        qrys_in = [data['qry_ids'] for data in batch_data]  # Assuming each query now has a unique ID
        qrys_in_seq_max_len = max(len(qrys) for qrys in qrys_in)
        qrys_in = pad_sequence([torch.tensor(qrys).long() for qrys in qrys_in], 
                             batch_first=True, 
                             padding_value=padding_value)
        qrys_in_mask = (qrys_in != padding_value).unsqueeze(-2)


        return 'search', {'uid': uids,
                          'pids_in': pids_in,
                          'pids_mask': pids_mask,
                          'pid_pred': pid_pred,
                          'pid_last_idx': pid_last_idx,
                          'qrys_in': qrys_in,
                          'qrys_in_mask': qrys_in_mask}


class BatchSampler(Sampler):
    """
    BatchSampler 类用于对不同类型的数据进行批量采样。

    参数:
    - data_source: 数据源，包含所有待采样的数据。
    - batch_size: 每个批次的数据量。

    属性:
    - recommendation_data_len: 存储推荐系统数据的长度和索引。
    - search_data_len: 存储搜索引擎数据的长度和索引。
    - batch_size: 每个批次的数据量。
    - pool_size: 用于计算的池子大小，默认为batch_size的100倍。
    """

    def __init__(self, data_source, batch_size):
        # 初始化父类
        super(BatchSampler, self).__init__(data_source)

        # 初始化数据长度列表
        self.recommendation_data_len = []
        self.search_data_len = []

        # 初始化batch_size和pool_size
        self.batch_size = batch_size
        self.pool_size = batch_size * 100

        # 遍历数据源，根据数据类型将数据长度和索引分别存储
        for idx, data in enumerate(data_source):
            if data['flag'] == 'recommendation':
                self.recommendation_data_len.append((idx, data['length']))
            else:  # data['flag'] == 'search'
                self.search_data_len.append((idx, data['length']))

    def __iter__(self):
        """
        生成批次索引的迭代器。根据数据类型分别生成批次，并对所有批次进行随机打乱。
        """
        # 初始化批次索引列表
        batch_idxes_list = []

        # 对推荐数据生成批次
        if self.recommendation_data_len:
            # 解压原始索引和长度
            recommendation_ori_idx, recommendation_len = zip(*self.recommendation_data_len)
            # 结合长度、随机排列的索引和原始索引，并按长度和池子编号排序
            recommendation_idx = zip(recommendation_len, np.random.permutation(len(self.recommendation_data_len)),
                                     recommendation_ori_idx)
            recommendation_idx = sorted(recommendation_idx, key=lambda e: (e[1] // self.pool_size, e[0]), reverse=True)
            # 按批次大小分割并添加到批次索引列表
            for i in range(0, len(recommendation_idx), self.batch_size):
                batch_idxes_list.append(
                    [recommendation_idx_[2] for recommendation_idx_ in recommendation_idx[i:i + self.batch_size]])

        # 对搜索数据生成批次
        if self.search_data_len:
            # 解压原始索引和长度
            search_ori_idx, search_len = zip(*self.search_data_len)
            # 结合长度、随机排列的索引和原始索引，并按长度和池子编号排序
            search_idx = zip(search_len, np.random.permutation(len(self.search_data_len)), search_ori_idx)
            search_idx = sorted(search_idx, key=lambda e: (e[1] // self.pool_size, e[0]), reverse=True)
            # 按批次大小分割并添加到批次索引列表
            for i in range(0, len(search_idx), self.batch_size):
                batch_idxes_list.append([search_idx_[2] for search_idx_ in search_idx[i:i + self.batch_size]])

        # 打乱批次顺序
        random.shuffle(batch_idxes_list)

        # 逐个返回批次索引
        for batch_idxes in batch_idxes_list:
            yield batch_idxes

    def __len__(self):
        """
        计算并返回批次的数量。

        返回:
        - int: 所有批次的数量。
        """
        # 计算推荐数据的批次数量
        recommendation_batches = (len(self.recommendation_data_len) + self.batch_size - 1) // self.batch_size
        # 计算搜索数据的批次数量
        search_batches = (len(self.search_data_len) + self.batch_size - 1) // self.batch_size
        # 返回总的批次数量
        return recommendation_batches + search_batches


if __name__ == "__main__":
    # for test only
    pass
