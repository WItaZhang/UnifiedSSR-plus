from modules import *
from copy import deepcopy, copy

def get_model(args):
    """
    根据参数构建统一的SSR模型。

    参数:
    - args: 包含所有模型相关参数的对象。

    返回:
    - model: 构建的SSR模型实例。
    """
    # 实例化UnifiedSSR模型
    model = UnifiedSSR(u_vocab=args.user_vocab,
                       p_vocab=args.product_vocab,
                       t_vocab=args.term_vocab,
                       q_vocab=args.query_vocab,
                       emb_size=args.emb_size,
                       hid_size=args.hid_size,
                       sub_seq_num=args.sub_seq_num,
                       enc_num_layer=args.enc_num_layer,
                       num_head=args.num_head,
                       tasks=args.tasks,
                       dropout=args.dropout)
    return model


class UnifiedSSR(nn.Module):
    def __init__(self, u_vocab, p_vocab, t_vocab, q_vocab, emb_size, hid_size, sub_seq_num, enc_num_layer,
                 num_head, tasks, padding_value=0, dropout=0.1):
        """
        UnifiedSSR模型的初始化函数。

        参数:
        - u_vocab (int): 用户词汇表大小。
        - p_vocab (int): 产品词汇表大小。
        - t_vocab (int): 标签词汇表大小。
        - emb_size (int): 嵌入层的维度。
        - hid_size (int): 隐藏层的维度，如果未提供，则默认为emb_size * 2。
        - sub_seq_num (int): 子序列的数量。
        - enc_num_layer (int): 编码器层数。
        - num_head (int): 多头注意力机制中的头数。
        - tasks (List[str]): 任务列表，用于模型训练和预测。
        - padding_value (int, 可选): 用于填充的值，默认为0。
        - dropout (float, 可选): Dropout的概率，默认为0.1。

        UnifiedSSR模型整合了用户、产品和标签的嵌入，通过多头注意力机制和编码器层来捕捉序列中的复杂关系。
        """
        super(UnifiedSSR, self).__init__()
        # 如果hid_size未提供，则设置为emb_size的两倍
        hid_size = hid_size or emb_size * 2
        self.tasks = tasks
        self.p_vocab = p_vocab
        self.t_vocab = t_vocab
        self.sub_seq_num = sub_seq_num
        self.emb_size = emb_size
        self.padding_value = padding_value
        # 初始化用户嵌入层
        self.u_embed = Embeddings(u_vocab, emb_size)
        # 初始化产品嵌入层
        self.p_embed = Embeddings(p_vocab, emb_size)
        # 初始化位置编码
        self.position = PositionalEncoding(emb_size, dropout)
        # 初始化标签嵌入层，包括开始和结束标记，因此维度是t_vocab + 2
        self.q_embed = Embeddings(q_vocab, emb_size)
        # 初始化编码器，使用Siamese结构
        self.encoder = SiameseEncoder(SiameseEncoderLayer(emb_size, hid_size, num_head, dropout), enc_num_layer)
        # 初始化序列分割模块，用于处理子序列
        self.seq_partition = SequencePartition(sub_seq_num, emb_size)
        # 初始化下一个产品搜索的权重参数
        self.next_product_search_w = nn.Parameter(torch.tensor(0.5))
        # 初始化标签权重参数
        self.q_t_w = torch.tensor(0.9)
        # 初始化损失函数
        self.loss = None


    def forward(self, task, inputs):
        """
        根据任务类型执行前向传播，返回相应的编码结果。
    
        Args:
            task (str): 任务类型，可以是'recommendation'或'search'。
            inputs (dict): 输入数据，包含不同任务类型所需的信息。
    
        Returns:
            p_enc (torch.Tensor): 产品编码，形状为[BS, Seq Max Len, Emb Size]。
            q_enc (torch.Tensor): 仅在任务为'search'时返回，表示查询编码，形状为[BS, Seq Max Len, Emb Size]。
        """
        # 根据任务类型设置相应的损失函数
        if task == 'recommendation':
            self.loss = self.next_product_predict_loss
            # 对产品ID进行嵌入并加上用户嵌入，然后通过位置编码处理
            p_rep = self.position(self.p_embed(inputs['pids_in']) + self.u_embed(inputs['uid']).unsqueeze(1))
            # 使用编码器对产品信息进行编码
            p_enc = self.encoder(p_rep, p_rep,p_rep, inputs['pids_mask'],0)
            return p_enc
        else:  # task == 'search'
            self.loss = self.next_product_search_loss
            p_rep = self.position(self.p_embed(inputs['pids_in']) + self.u_embed(inputs['uid']).unsqueeze(1))
            q_rep = self.q_embed(inputs['qrys_in'])
            q_rep = self.position(q_rep + self.u_embed(inputs['uid']).unsqueeze(1))
            q_rep, q_last = (q_rep[:, :-1, :]).clone(), (q_rep[:, 1:, :]).clone()
            q_mask_input, q_last_mask_input = (inputs['qrys_in_mask'][:, :, :-1]).clone(), inputs['qrys_in_mask'][:, :, 1:]

            p_enc = self.encoder(p_rep, q_rep,p_rep, inputs['pids_mask'],0)
            q_enc = self.encoder(q_rep, p_rep, q_last, q_mask_input,1)
         
            return p_enc, q_enc, q_last

    def next_product_predict_loss(self, seq_emb, mask, p_pos, p_negs):
        """
        计算下一个商品预测的损失函数。
        
        该函数通过比较正样本和负样本的嵌入与序列嵌入的相似度来预测下一个商品，
        并使用sigmoid交叉熵损失函数来计算损失。
        
        参数:
        - seq_emb: 序列嵌入，形状为[BS*MaxLen, EmbSize]。
        - mask: 掩码，用于区分有效和填充部分，形状为[BS*MaxLen]。
        - p_pos: 正样本商品ID，形状为[BS*MaxLen]。
        - p_negs: 负样本商品ID，形状为[BS*MaxLen, NumNeg]。
        
        返回:
        - loss: 预测损失。
        """

        # 正样本商品嵌入
        p_pos_emb = self.p_embed(p_pos)
        # p_pos_emb [BS*MaxLen, EmbSize]

        # 计算正样本的logits
        p_pos_logits = torch.sum(p_pos_emb * seq_emb, -1)
        # p_pos_logits [BS*MaxLen]

        # 负样本商品嵌入
        p_negs_emb = self.p_embed(p_negs)
        # p_negs_emb [BS*MaxLen, NumNeg, EmbSize]

        # 计算负样本的logits
        p_negs_logits = torch.sum(p_negs_emb * seq_emb.unsqueeze(1).repeat(1, p_negs_emb.size(1), 1), -1)
        # p_negs_logits [BS*MaxLen, NumNeg]

        # 计算损失函数
        loss = - torch.sum(
            torch.log(p_pos_logits.sigmoid() + 1e-24) * mask +
            torch.log(1 - p_negs_logits.sigmoid() + 1e-24).sum(-1) * mask
        ) / mask.sum()

        return loss

    def next_product_predict(self, seq_emb, last_idx, p_pred=None):
        """
        根据序列嵌入和最后一个元素的索引预测下一个产品。
        
        参数:
        - seq_emb: 序列嵌入，形状为[BS, SeqLen, EmbSize]。
        - last_idx: 序列中最后一个元素的索引，形状为[BS]。
        - p_pred: 可选，预测的产品索引，形状为[BS, NumNeg+1]。
        
        返回:
        - pred_logits: 预测的 logits，形状为[BS, NumNeg+1]或[BS, PVocab]，取决于p_pred是否为None。
        """
        # 为last_idx添加维度以便进行后续操作
        last_idx = last_idx.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, seq_emb.size(-1))
        # 从序列嵌入中提取最后一个元素的嵌入
        seq_last_out = seq_emb.gather(1, last_idx).squeeze(1)
        if p_pred is not None:
            # 如果提供了p_pred，则计算这些特定产品的预测logits
            p_emb = self.p_embed(p_pred)
            # p_emb [BS, NumNeg+1, EmbSize]
            seq_last_out = seq_last_out.unsqueeze(1).repeat(1, p_emb.size(1), 1)
            # seq_last_out [BS, NumNeg+1 or PVocab, EmbSize]
            pred_logits = torch.sum(p_emb * seq_last_out, -1)
            # pred_logits [BS, NumNeg+1 or PVocab]
            return pred_logits
        else:
            # 如果没有提供p_pred，则计算所有可能产品的预测logits
            p_emb = self.p_embed.lut.weight * math.sqrt(self.emb_size)
            # p_emb = self.p_embed.lut.weight
            # p_emb [PVocab, EmbSize]
            p_emb = p_emb.unsqueeze(0).repeat(seq_emb.size(0), 1, 1)
            # p_emb [BS, PVocab, EmbSize]
            # 由于产品词汇量可能非常大，因此分批处理以避免内存问题
            p_emb_chunks = [p_emb[:, i:i + 5000] for i in range(0, p_emb.size(1), 5000)]
            pred_logits = []
            for p_emb_chunk in p_emb_chunks:
                # 对每个产品嵌入分块计算预测logits
                if p_emb.device.type == 'cuda':
                    pred_logits.append(
                        torch.sum(p_emb_chunk * seq_last_out.unsqueeze(1).repeat(1, p_emb_chunk.size(1), 1), -1).cpu())
                else:
                    pred_logits.append(
                        torch.sum(p_emb_chunk * seq_last_out.unsqueeze(1).repeat(1, p_emb_chunk.size(1), 1), -1))
            # 合并所有分块的预测logits
            return torch.cat(pred_logits, dim=1)

    def next_product_search_loss(self, p_seq_emb, q_seq_emb, q_last_emb, mask, p_pos, p_negs):
        """
        计算下一个商品搜索的损失。
    
        该方法旨在通过比较正样本和负样本的嵌入与查询序列嵌入之间的相似度来优化模型参数。
        
        参数:
        - p_seq_emb: [BS*MaxLen, EmbSize] 商品序列嵌入。
        - q_seq_emb: [BS*MaxLen, EmbSize] 查询序列嵌入。
        - mask: [BS*MaxLen] 掩码，用于区分有效和填充部分。
        - p_pos: [BS*MaxLen] 正样本商品ID。
        - p_negs: [BS*MaxLen, NumNeg] 负样本商品ID。
    
        返回:
        - loss: 经过加权的损失值。
        """

        # 正样本商品嵌入
        p_pos_emb = self.p_embed(p_pos)
        # p_pos_emb [BS*MaxLen, EmbSize]

        # 负样本商品嵌入
        p_negs_emb = self.p_embed(p_negs)
        # p_negs_emb [BS*MaxLen, NumNeg, EmbSize]

        # 计算正样本与商品序列嵌入的相似度得分
        p_pos_sc = torch.sum(p_pos_emb * p_seq_emb, -1)
        # p_pos_sc [BS*MaxLen]

        # 计算负样本与商品序列嵌入的相似度得分
        p_negs_sc = torch.sum(p_negs_emb * p_seq_emb.unsqueeze(1).repeat(1, p_negs_emb.size(1), 1), -1)
        # p_negs_sc [BS*MaxLen, NumNeg]

        # 计算商品搜索的损失
        p_loss = - torch.sum(
            torch.log(p_pos_sc.sigmoid() + 1e-24) * mask +
            torch.log(1 - p_negs_sc.sigmoid() + 1e-24).sum(-1) * mask
        ) / mask.sum()

        # 计算正样本与查询序列嵌入的相似度得分
        q_pos_sc = torch.sum((p_pos_emb * q_seq_emb), -1)
        # q_pos_sc [BS*MaxLen]
        q_negs_sc = torch.sum((p_negs_emb * q_seq_emb.unsqueeze(1).repeat(1, p_negs_emb.size(1), 1)), -1)

        q_t_sc = torch.sum((p_pos_emb * q_last_emb), -1)
        q_t_negs_sc = torch.sum((p_negs_emb * q_last_emb.unsqueeze(1).repeat(1, p_negs_emb.size(1), 1)), -1)
        # 计算负样本与查询序列嵌入的相似度得分

        # q_negs_sc [BS*MaxLen,NumNeg]

        # 计算查询搜索的损失
        q_loss = - torch.sum(
            torch.log(q_pos_sc.sigmoid() + 1e-24) * mask +
            torch.log(1 - q_negs_sc.sigmoid() + 1e-24).sum(-1) * mask
        ) / mask.sum()

        q_last_loss = - torch.sum(
            torch.log(q_t_sc.sigmoid() + 1e-24) * mask +
            torch.log(1 - q_t_negs_sc.sigmoid() + 1e-24).sum(-1) * mask
        ) / mask.sum()

        # 限制next_product_search_w的范围在[0.1, 0.9]之间，以确保稳定性和有效性
        self.next_product_search_w.data = self.next_product_search_w.clamp(min=0.1, max=0.9)
        # self.q_t_w.data = self.q_t_w.clamp(min=0.1, max=0.9)

        # 返回加权后的总损失
        return (1 - self.q_t_w) * (self.next_product_search_w * p_loss + (1 - self.next_product_search_w) * q_loss) + self.q_t_w * q_last_loss

    def next_product_search(self, p_seq_emb, q_seq_emb, q_last_emb, last_idx, p_pred=None):
        """
        根据给定的序列嵌入和最后一个产品索引，进行下一个产品搜索。
        
        参数:
        - p_seq_emb: 购买序列的嵌入表示 [BS, SeqLen, EmbSize]
        - q_seq_emb: 搜索序列的嵌入表示 [BS, SeqLen, EmbSize]
        - last_idx: 序列中最后一个产品的索引 [BS]
        - p_pred: 下一个可能购买的产品预测 [BS, NumNeg+1]
        
        返回:
        - 下一个产品搜索的预测分数 [BS, PVocab] 或 [BS, NumNeg+1]
        """
        # 扩展 last_idx 以匹配 p_seq_emb 的维度
        last_idx = last_idx.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, p_seq_emb.size(-1))
        # 从 p_seq_emb 中提取最后一个产品的嵌入表示
        p_seq_last_out = p_seq_emb.gather(1, last_idx).squeeze()  # [BS, EmbSize]
        # 从 q_seq_emb 中提取最后一个产品的嵌入表示
        q_seq_last_out = q_seq_emb.gather(1, last_idx).squeeze()  # [BS, EmbSize]
        q_seq_last_t = q_last_emb.gather(1, last_idx).squeeze()

        if p_pred is not None:
            # 如果提供了 p_pred，则计算其嵌入表示
            p_emb = self.p_embed(p_pred)
            # p_emb [BS, NumNeg+1, EmbSize]

            # 扩展 p_seq_last_out 和 q_seq_last_out 以匹配 p_emb 的维度
            p_seq_last_out = p_seq_last_out.unsqueeze(1).repeat(1, p_emb.size(1), 1)
            # p_seq_last_out [BS, NumNeg+1, EmbSize]
            q_seq_last_out = q_seq_last_out.unsqueeze(1).repeat(1, p_emb.size(1), 1)
            q_seq_last_t = q_seq_last_t.unsqueeze(1).repeat(1, p_emb.size(1), 1)
            # q_seq_last_out [BS, NumNeg+1, EmbSize]
            # 计算购买序列的预测分数
            p_pred_logits = torch.sum(p_emb * p_seq_last_out, -1)  # [BS, NumNeg+1]
            # 计算搜索序列的预测分数
            q_pred_logits = torch.sum(p_emb * q_seq_last_out, -1)  # [BS, NumNeg+1]
            q_last_pred_logits = torch.sum(p_emb * q_seq_last_t, -1)

            # 结合购买和搜索序列的预测分数
            return ((1 - self.q_t_w) * (self.next_product_search_w * p_pred_logits +
                    (1 - self.next_product_search_w) * q_pred_logits) +
                    self.q_t_w * q_last_pred_logits)
        else:
            # 如果没有提供 p_pred，则使用所有产品词汇的嵌入表示
            p_emb = self.p_embed.lut.weight * math.sqrt(self.emb_size)  # [PVocab, EmbSize]
            # p_emb = self.p_embed.lut.weight # [PVocab, EmbSize]
            # 扩展 p_emb 以匹配 p_seq_emb 的批次大小
            p_emb = p_emb.unsqueeze(0).repeat(p_seq_emb.size(0), 1, 1)
            # p_emb [BS, PVocab, EmbSize]
            # 将 p_emb 分块处理以减少内存消耗
            p_emb_chunks = [p_emb[:, i:i + 5000] for i in range(0, p_emb.size(1), 5000)]
            pred_logits = []
            for p_emb_chunk in p_emb_chunks:
                # 计算购买序列的预测分数
                p_pred_logits_ = torch.sum(
                    p_emb_chunk * p_seq_last_out.unsqueeze(1).repeat(1, p_emb_chunk.size(1), 1), -1)
                # 计算搜索序列的预测分数
                q_pred_logits_ = torch.sum(
                    p_emb_chunk * q_seq_last_out.unsqueeze(1).repeat(1, p_emb_chunk.size(1), 1), -1)
                q_last_pred_logits_ = torch.sum(
                    p_emb_chunk * q_seq_last_t.unsqueeze(1).repeat(1, p_emb_chunk.size(1), 1), -1)
                q_pred_logits_ = (1 - self.q_t_w) * q_pred_logits_ + self.q_t_w * q_last_pred_logits_
                # 结合购买和搜索序列的预测分数
                pred_logits_ = self.next_product_search_w * p_pred_logits_ + (
                        1 - self.next_product_search_w) * q_pred_logits_
                # 将预测分数添加到列表中
                if p_emb.device.type == 'cuda':
                    pred_logits.append(pred_logits_.cpu())
                else:
                    pred_logits.append(pred_logits_)
            # 合并所有分块的预测分数
            return torch.cat(pred_logits, dim=1)

    def get_sub_seq_wins(self, emb,last_idx):
        sub_seq_wins = self.seq_partition(emb,last_idx)  # [BS, Sub Seq Num, 2]
        return sub_seq_wins
    
    def intra_corr_loss(self, emb, sub_seq_wins, mask):
        len_idx = torch.arange(emb.size(1), device=emb.device).unsqueeze(0)  # [1, Seq Max Len]
        sub_mask = (sub_seq_wins[:, :, 0:1] <= len_idx) & (len_idx <= sub_seq_wins[:, :, 1:2])
        sub_mask = sub_mask & mask
        # sub_mask [BS, Sub Seq Num, Seq Max Len]
        sub_mask = sub_mask.unsqueeze(-1).expand(-1, -1, -1, emb.size(-1))
        # sub_mask [BS, Sub Seq Num, Seq Max Len, Emb Size]
        sub_seq_rep = emb.unsqueeze(1) * sub_mask.float()
        emb = emb + sub_seq_rep.sum(dim=1) / (sub_mask.sum(dim=1) + 1e-10)

        sub_seq_rep = sub_seq_rep.sum(dim=-2) / (sub_mask.sum(dim=-2) + 1e-10)
        intra_corr = F.cosine_similarity(sub_seq_rep.unsqueeze(2), sub_seq_rep.unsqueeze(1), dim=-1)
        intra_corr = torch.abs(intra_corr)
        corr_mask = torch.triu(torch.ones((1, sub_seq_rep.size(1), sub_seq_rep.size(1)), device=sub_seq_rep.device),
                               diagonal=1).bool()
        corr_mask = corr_mask & ~torch.triu(
            torch.ones((1, sub_seq_rep.size(1), sub_seq_rep.size(1)), device=sub_seq_rep.device), diagonal=2).bool()
        intra_corr = intra_corr * corr_mask.float()
        intra_corr_loss = intra_corr.sum() / (intra_corr.nonzero().size(0) + 1e-10)
        return emb, sub_seq_rep, intra_corr_loss

    def inter_corr_loss(self, p_emb, p_sub_seq_wins, q_emb, q_sub_seq_wins, mask):
        p_emb, p_sub_seq_rep, p_intra_corr_loss = self.intra_corr_loss(p_emb, p_sub_seq_wins, mask)
        q_emb, q_sub_seq_rep, q_intra_corr_loss = self.intra_corr_loss(q_emb, q_sub_seq_wins, mask)
        inter_corr = F.cosine_similarity(p_sub_seq_rep, q_sub_seq_rep, dim=-1)
        inter_corr_loss = inter_corr.sum() / (inter_corr.nonzero().size(0) + 1e-10)
        inter_corr_loss = p_intra_corr_loss + q_intra_corr_loss - inter_corr_loss
        return p_emb, q_emb, inter_corr_loss