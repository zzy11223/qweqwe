import argparse
import torch
import torch.nn as nn
from dgl.dataloading import GraphDataLoader

from dataloader import MixTrafficFlowDataset4DGL
from model_new_aug import MixTemporalGNN_SSL
from optim import GradualWarmupScheduler
from utils import show_time, set_seed, get_device, mix_collate_cl_fn
from config import *

torch.autograd.set_detect_anomaly(True)

def pretrain():
    model = MixTemporalGNN_SSL(num_classes=config.NUM_CLASSES, embedding_size=config.EMBEDDING_SIZE, h_feats=config.H_FEATS,
                           dropout=config.DROPOUT, downstream_dropout=config.DOWNSTREAM_DROPOUT, point=opt.point,
                           seq_aug_ratio=opt.seq_aug_ratio, drop_edge_ratio=opt.drop_edge_ratio,
                           drop_node_ratio=opt.drop_node_ratio, K=opt.K, hp_ratio=opt.hp_ratio, tau=opt.tau, gtau=opt.gtau)
    
    dataset = MixTrafficFlowDataset4DGL(header_path=config.HEADER_TRAIN_GRAPH_DATA,
                                      payload_path=config.TRAIN_GRAPH_DATA,
                                      point=opt.point,
                                      perc=opt.perc)
    
    dataloader = GraphDataLoader(dataset, batch_size=config.BATCH_SIZE if opt.bs == -1 else opt.bs, 
                               shuffle=True, collate_fn=mix_collate_cl_fn,
                               num_workers=num_workers, pin_memory=True)
    
    model = model.to(device)
    model.train()
    
    num_steps = len(dataloader) * config.PRETRAIN_EPOCH
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, 
                                                         T_max=num_steps - int(num_steps * config.WARM_UP), 
                                                         eta_min=config.LR_MIN)
    warmup_scheduler = GradualWarmupScheduler(optimizer, 
                                            warmup_iter=int(num_steps * config.WARM_UP), 
                                            after_scheduler=scheduler)
    warmup_scheduler.step()  # Warm up starts from lr = 0
    
    for epoch in range(config.PRETRAIN_EPOCH):
        loss_all = []
        graph_loss_all = []
        seq_loss_all = []
        
        for batch_id, (header_data, payload_data, _, header_mask, payload_mask) in enumerate(dataloader):
            header_data = header_data.to(device, non_blocking=True)
            payload_data = payload_data.to(device, non_blocking=True)

            #header_mask = header_mask.to(device, non_blocking=True)
            #payload_mask = payload_mask.to(device, non_blocking=True)
            
            # 只关注对比学习
            _, seq_cl_loss, graph_cl_loss = model.forward_pretrain(header_data, payload_data, header_mask, payload_mask)
            
            # 总损失 - 不使用标签
            loss = opt.coe * seq_cl_loss + opt.coe_graph * graph_cl_loss
            
            loss_all.append(float(loss))
            seq_loss_all.append(float(seq_cl_loss))
            graph_loss_all.append(float(graph_cl_loss))
            
            loss /= (config.GRADIENT_ACCUMULATION if opt.ga == -1 else opt.ga)
            loss.backward()
            
            if ((batch_id + 1) % (config.GRADIENT_ACCUMULATION if opt.ga == -1 else opt.ga) == 0) or (batch_id + 1 == len(dataloader)):
                optimizer.step()
                optimizer.zero_grad()
            
            warmup_scheduler.step()
            
            if batch_id % 10 == 0:
                print('{} In epoch {}, batch {}, lr: {:.5f}, total_loss: {:.4f}, seq_loss: {:.4f}, graph_loss: {:.4f}'.format(
                    show_time(), epoch, batch_id, optimizer.param_groups[0]['lr'], float(loss), 
                    float(seq_cl_loss), float(graph_cl_loss)))
        
        # 每个epoch结束后打印平均损失
        avg_loss = sum(loss_all) / len(loss_all)
        avg_seq_loss = sum(seq_loss_all) / len(seq_loss_all)
        avg_graph_loss = sum(graph_loss_all) / len(graph_loss_all)
        
        print('{} Epoch {} finished, avg_loss: {:.4f}, avg_seq_loss: {:.4f}, avg_graph_loss: {:.4f}'.format(
            show_time(), epoch, avg_loss, avg_seq_loss, avg_graph_loss))

    # 保存预训练模型
    torch.save(model.state_dict(), config.MIX_MODEL_CHECKPOINT[:-4] + '_pretrained_' + str(opt.prefix) + '.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="dataset", required=True)
    parser.add_argument("--num_workers", type=int, help="num workers", default=-1)
    parser.add_argument("--prefix", type=str, required=True)
    parser.add_argument("--point", type=int, default=15)
    parser.add_argument("--coe", type=float, default=0.5)
    parser.add_argument("--coe_graph", type=float, default=1.0)
    parser.add_argument("--seq_aug_ratio", type=float, default=0.6)
    parser.add_argument("--tau", type=float, default=0.07)
    parser.add_argument("--gtau", type=float, default=0.04)
    parser.add_argument("--drop_edge_ratio", type=float, default=0.05)
    parser.add_argument("--drop_node_ratio", type=float, default=0.1)
    parser.add_argument("--K", type=int, default=15)
    parser.add_argument("--hp_ratio", type=float, default=0.5)
    parser.add_argument("--perc", type=float, default=1.0)
    parser.add_argument("--bs", type=int, default=-1)
    parser.add_argument("--ga", type=int, default=-1)
    opt = parser.parse_args()

    if opt.dataset == 'iscx-vpn':
        config = ISCXVPNConfig()
    elif opt.dataset == 'iscx-nonvpn':
        config = ISCXNonVPNConfig()
    elif opt.dataset == 'iscx-tor':
        config = ISCXTorConfig()
    elif opt.dataset == 'iscx-nontor':
        config = ISCXNonTorConfig()
    else:
        raise Exception('Dataset Error')

    # 添加预训练epoch
    config.PRETRAIN_EPOCH = 50  # 可调整

    device = get_device(index=0)
    num_workers = 0
    set_seed()
    pretrain()