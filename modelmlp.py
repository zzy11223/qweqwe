import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dgl.nn.pytorch.conv import SAGEConv


class GCN(nn.Module):
    def __init__(self, embedding_size, h_feats, dropout):
        super(GCN, self).__init__()
        self.gcn_out_dim = 4 * h_feats
        self.embedding = nn.Embedding(256 + 1, embedding_size)
        self.gcn1 = SAGEConv(embedding_size, h_feats, 'mean', feat_drop=dropout, activation=nn.PReLU(h_feats), norm=nn.BatchNorm1d(h_feats))
        self.gcn2 = SAGEConv(h_feats, h_feats, 'mean', feat_drop=dropout, activation=nn.PReLU(h_feats), norm=nn.BatchNorm1d(h_feats))
        self.gcn3 = SAGEConv(h_feats, h_feats, 'mean', feat_drop=dropout, activation=nn.PReLU(h_feats), norm=nn.BatchNorm1d(h_feats))
        self.gcn4 = SAGEConv(h_feats, h_feats, 'mean', activation=nn.PReLU(h_feats), norm=nn.BatchNorm1d(h_feats))

    def forward(self, g, in_feat):
        in_feat = in_feat.long()
        h = self.embedding(in_feat.view(-1))
        h1 = self.gcn1(g, h)
        h2 = self.gcn2(g, h1)
        h3 = self.gcn3(g, h2)
        h4 = self.gcn4(g, h3)
        g.ndata['h'] = torch.cat((h1, h2, h3, h4), dim=1)
        g_vec = dgl.mean_nodes(g, 'h')

        return g_vec


class Cross_Gated_Info_Filter(nn.Module):
    def __init__(self, in_size, point):
        super(Cross_Gated_Info_Filter, self).__init__()
        self.filter1 = nn.Sequential(
            nn.Linear(in_size, in_size),
            nn.PReLU(point),
            nn.Linear(in_size, in_size)
        )
        self.filter2 = nn.Sequential(
            nn.Linear(in_size, in_size),
            nn.PReLU(point),
            nn.Linear(in_size, in_size)
        )

    def forward(self, x, y):
        ori_x = x
        ori_y = y
        z1 = self.filter1(x).sigmoid() * ori_y
        z2 = self.filter2(y).sigmoid() * ori_x

        return torch.cat([z1, z2], dim=-1)


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class MLP_Projection(nn.Module):
    """MLP projection head for contrastive learning"""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MLP_Projection, self).__init__()
        self.layer1 = nn.Linear(in_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x


class MixTemporalGNN_SSL(nn.Module):
    def __init__(self, num_classes, embedding_size=64, h_feats=128, dropout=0.2, downstream_dropout=0.0,
                 point=15, seq_aug_ratio=0.8, drop_edge_ratio=0.1, drop_node_ratio=0.1, K=15,
                 hp_ratio=0.5, tau=0.07, gtau=0.07, projection_dim=256):
        super(MixTemporalGNN_SSL, self).__init__()
        self.header_graphConv = GCN(embedding_size=embedding_size, h_feats=h_feats, dropout=dropout)
        self.payload_graphConv = GCN(embedding_size=embedding_size, h_feats=h_feats, dropout=dropout)
        self.gcn_out_dim = 4 * h_feats
        self.point = point
        self.seq_aug_ratio = seq_aug_ratio
        self.drop_edge_ratio = drop_edge_ratio
        self.drop_node_ratio = drop_node_ratio
        self.K = K
        self.hp_ratio = hp_ratio
        self.gated_filter = Cross_Gated_Info_Filter(in_size=self.gcn_out_dim, point=self.point)
        self.rnn = nn.LSTM(input_size=self.gcn_out_dim * 2, hidden_size=self.gcn_out_dim * 2, num_layers=2, bidirectional=True, dropout=downstream_dropout)
        
        # 预训练和微调阶段都会用到的编码器部分
        self.encoder = nn.Sequential(
            nn.Linear(in_features=self.gcn_out_dim * 4, out_features=self.gcn_out_dim),
            nn.PReLU(self.gcn_out_dim)
        )
        
        # 新增：MLP投影头，用于对比学习
        self.seq_projector = MLP_Projection(in_dim=self.gcn_out_dim, 
                                          hidden_dim=self.gcn_out_dim//2, 
                                          out_dim=projection_dim)
        
        self.graph_projector = MLP_Projection(in_dim=self.gcn_out_dim, 
                                           hidden_dim=self.gcn_out_dim//2, 
                                           out_dim=projection_dim)
        
        # 分类器，仅在微调阶段使用
        self.classifier = nn.Linear(in_features=self.gcn_out_dim, out_features=num_classes)
        
        # 包级别的分类器，仅在微调阶段使用
        self.packet_head = nn.Sequential(
            nn.Linear(in_features=self.gcn_out_dim * 2, out_features=self.gcn_out_dim),
            nn.PReLU(self.gcn_out_dim),
            nn.Linear(in_features=self.gcn_out_dim, out_features=num_classes)
        )

        # 对比学习相关
        self.supcl = SupConLoss(temperature=tau, base_temperature=tau)
        self.supcl_g = SupConLoss(temperature=gtau, base_temperature=gtau)
        self.drop_edge_trans = dgl.DropEdge(p=self.drop_edge_ratio)
        self.drop_node_trans = dgl.DropNode(p=self.drop_node_ratio)
    
    def get_representations(self, header_graph_data, payload_graph_data,  header_mask, payload_mask):
        """获取表示，不包含分类任务"""

        header_mask = header_mask.reshape(header_mask.shape[0] // self.point, self.point, -1)[:, :self.K, :].reshape(-1)
        payload_mask = payload_mask.reshape(header_mask.shape[0] // self.point, self.point, -1)[:, :self.K, :].reshape(-1)
        
        # 原始数据处理
        header_gcn_out = self.header_graphConv(header_graph_data, header_graph_data.ndata['feat']).reshape(
            -1, self.point, self.gcn_out_dim)
        payload_gcn_out = self.payload_graphConv(payload_graph_data, payload_graph_data.ndata['feat']).reshape(
            -1, self.point, self.gcn_out_dim)
        
        # 融合表示
        gcn_out = self.gated_filter(header_gcn_out, payload_gcn_out)
        
        # RNN处理序列
        gcn_out_transposed = gcn_out.transpose(0, 1)
        _, (h_n, _) = self.rnn(gcn_out_transposed)
        rnn_out = torch.cat((h_n[-1], h_n[-2]), dim=1)
        
        # 获取最终表示
        representations = self.encoder(rnn_out)
        
        return representations, header_gcn_out, payload_gcn_out, gcn_out, header_mask, payload_mask

    def forward_pretrain(self, header_graph_data, payload_graph_data, header_mask, payload_mask):
        """预训练阶段前向传播：仅包含对比学习"""
        batch_size = header_mask.shape[0] // self.point
        
        # 获取原始表示
        representations, header_gcn_out, payload_gcn_out, gcn_out, header_mask, payload_mask = self.get_representations(
            header_graph_data, payload_graph_data, header_mask, payload_mask)
        
        # 图增强
        aug_header_graph_data = self.drop_node_trans(self.drop_edge_trans(header_graph_data))
        aug_payload_graph_data = self.drop_node_trans(self.drop_edge_trans(payload_graph_data))
        
        # 获取增强图的表示
        aug_header_gcn_out = self.header_graphConv(aug_header_graph_data, aug_header_graph_data.ndata['feat']).reshape(
            batch_size, self.point, -1)
        aug_payload_gcn_out = self.payload_graphConv(aug_payload_graph_data, aug_payload_graph_data.ndata['feat']).reshape(
            batch_size, self.point, -1)
        
        # 计算图级对比损失
        temp1 = header_gcn_out[:, :self.K, :].reshape(-1, header_gcn_out.shape[2])[header_mask]
        temp2 = aug_header_gcn_out[:, :self.K, :].reshape(-1, aug_header_gcn_out.shape[2])[header_mask]
        mask12 = torch.any(temp1 != 0, dim=1) & torch.any(temp2 != 0, dim=1)
        temp3 = payload_gcn_out[:, :self.K, :].reshape(-1, payload_gcn_out.shape[2])[payload_mask]
        temp4 = aug_payload_gcn_out[:, :self.K, :].reshape(-1, aug_payload_gcn_out.shape[2])[payload_mask]
        mask34 = torch.any(temp3 != 0, dim=1) & torch.any(temp4 != 0, dim=1)
        
        # 使用投影头处理图表示
        
            # 应用图投影头
        projected_temp1 = self.graph_projector(temp1[mask12])
        projected_temp2 = self.graph_projector(temp2[mask12])
            
        header_cl_loss = self.supcl_g(torch.cat(
                (F.normalize(projected_temp1, p=2).unsqueeze(1), 
                 F.normalize(projected_temp2, p=2).unsqueeze(1)), dim=1))
            
        
            # 应用图投影头
        projected_temp3 = self.graph_projector(temp3[mask34])
        projected_temp4 = self.graph_projector(temp4[mask34])
            
        payload_cl_loss = self.supcl_g(torch.cat(
                (F.normalize(projected_temp3, p=2).unsqueeze(1), 
                 F.normalize(projected_temp4, p=2).unsqueeze(1)), dim=1))
    
            
        graph_cl_loss = self.hp_ratio * header_cl_loss + (1 - self.hp_ratio) * payload_cl_loss
        
        # 序列增强
        gcn_out_aug = self.gated_filter(aug_header_gcn_out, aug_payload_gcn_out)
        aug_index = []
        for _ in range(len(gcn_out_aug)):
            index = np.random.choice(range(self.point), size=int(self.point * self.seq_aug_ratio), replace=False)
            index.sort()
            aug_index.append(index)
        
        aug_index = torch.tensor(np.array(aug_index), dtype=int, device=gcn_out.device)
        aug_index = aug_index.unsqueeze(2)
        aug_index = aug_index.repeat(1, 1, gcn_out_aug.shape[2])
        gcn_out_aug = torch.gather(gcn_out_aug, dim=1, index=aug_index)
        
        # 处理增强序列
        gcn_out_aug = gcn_out_aug.transpose(0, 1)
        _, (h_n_aug, _) = self.rnn(gcn_out_aug)
        rnn_out_aug = torch.cat((h_n_aug[-1], h_n_aug[-2]), dim=1)
        
        # 获取增强序列的表示
        representations_aug = self.encoder(rnn_out_aug)
        
        # 应用序列投影头
        projected_rep = self.seq_projector(representations)
        projected_rep_aug = self.seq_projector(representations_aug)
        
        # 计算序列级对比损失
        seq_cl_loss = self.supcl(torch.cat(
            (F.normalize(projected_rep, p=2).unsqueeze(1), 
             F.normalize(projected_rep_aug, p=2).unsqueeze(1)), dim=1))
        
        return representations, seq_cl_loss, graph_cl_loss

    def forward_finetune(self, header_graph_data, payload_graph_data, labels, header_mask, payload_mask):
        """微调阶段前向传播：仅包含分类任务"""
        batch_size = header_mask.shape[0] // self.point
        
        # 获取表示
        representations, header_gcn_out, payload_gcn_out, gcn_out, header_mask, payload_mask = self.get_representations(
            header_graph_data, payload_graph_data, header_mask, payload_mask)
        
        # 包级别分类
        packet_mask = header_mask & payload_mask
        packet_rep = gcn_out[:, :self.K, :].reshape(-1, gcn_out.shape[2])[packet_mask]
        packet_label = labels.reshape(-1, 1).repeat(1, self.point)[:, :self.K].reshape(-1)[packet_mask]
        packet_out = self.packet_head(packet_rep)
        
        # 流级别分类
        out = self.classifier(representations)
        
        return out, packet_out, packet_label, representations