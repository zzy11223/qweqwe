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

def finetune():
    # 创建模型实例
    model = MixTemporalGNN_SSL(num_classes=config.NUM_CLASSES, embedding_size=config.EMBEDDING_SIZE, h_feats=config.H_FEATS,
                           dropout=config.DROPOUT, downstream_dropout=config.DOWNSTREAM_DROPOUT, point=opt.point,
                           seq_aug_ratio=opt.seq_aug_ratio, drop_edge_ratio=opt.drop_edge_ratio,
                           drop_node_ratio=opt.drop_node_ratio, K=opt.K, hp_ratio=opt.hp_ratio, tau=opt.tau, gtau=opt.gtau)
    
    # 加载预训练权重
    pretrained_weights = torch.load(config.MIX_MODEL_CHECKPOINT[:-4] + '_pretrained_' + str(opt.load_prefix) + '.pth')
    model.load_state_dict(pretrained_weights)
    
    # 数据加载
    dataset = MixTrafficFlowDataset4DGL(header_path=config.HEADER_TRAIN_GRAPH_DATA,
                                      payload_path=config.TRAIN_GRAPH_DATA,
                                      point=opt.point,
                                      perc=opt.perc)
    
    dataloader = GraphDataLoader(dataset, batch_size=config.BATCH_SIZE if opt.bs == -1 else opt.bs, 
                               shuffle=True, collate_fn=mix_collate_cl_fn,
                               num_workers=num_workers, pin_memory=True)
    
    model = model.to(device)
    model.train()
    
    # 冻结编码器部分（如果需要）
    if opt.freeze_encoder:
        for name, param in model.named_parameters():
            if 'header_graphConv' in name or 'payload_graphConv' in name or 'gated_filter' in name or 'rnn' in name or 'encoder' in name:
                param.requires_grad = False
    
    # 选择仅训练分类器部分的参数
    if opt.freeze_encoder:
        train_params = [p for n, p in model.named_parameters() if p.requires_grad]
    else:
        train_params = model.parameters()
    
    num_steps = len(dataloader) * config.FINETUNE_EPOCH
    optimizer = torch.optim.Adam(train_params, lr=config.FINETUNE_LR if hasattr(config, 'FINETUNE_LR') else config.LR, 
                               weight_decay=config.WEIGHT_DECAY)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, 
                                                         T_max=num_steps - int(num_steps * config.WARM_UP), 
                                                         eta_min=config.LR_MIN)
    
    warmup_scheduler = GradualWarmupScheduler(optimizer, 
                                            warmup_iter=int(num_steps * config.WARM_UP), 
                                            after_scheduler=scheduler)
    
    warmup_scheduler.step()  # Warm up starts from lr = 0
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    
    for epoch in range(config.FINETUNE_EPOCH):
        num_correct = 0
        num_tests = 0
        num_correct_packet = 0
        num_tests_packet = 0
        loss_all = []
        
        for batch_id, (header_data, payload_data, labels, header_mask, payload_mask) in enumerate(dataloader):
            header_data = header_data.to(device, non_blocking=True)
            payload_data = payload_data.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            #header_mask = header_mask.to(device, non_blocking=True)
            #payload_mask = payload_mask.to(device, non_blocking=True)
            
            # 微调阶段：专注于分类任务
            pred, packet_out, packet_label, _ = model.forward_finetune(
                header_data, payload_data, labels, header_mask, payload_mask)
            
            # 分类损失
            loss = criterion(pred, labels) +criterion(packet_out, packet_label)
            
            loss_all.append(float(loss))
            num_correct += (pred.argmax(1) == labels).sum().item()
            num_tests += len(labels)
            num_correct_packet += (packet_out.argmax(1) == packet_label).sum().item()
            num_tests_packet += len(packet_label)
            
            loss /= (config.GRADIENT_ACCUMULATION if opt.ga == -1 else opt.ga)
            loss.backward()
            
            if ((batch_id + 1) % (config.GRADIENT_ACCUMULATION if opt.ga == -1 else opt.ga) == 0) or (batch_id + 1 == len(dataloader)):
                optimizer.step()
                optimizer.zero_grad()
            
            warmup_scheduler.step()
            
            if batch_id % 10 == 0:
                print('{} In epoch {}, batch {}, lr: {:.5f}, loss: {:.4f}, acc: {:.3f}, acc_packet: {:.3f}'.format(
                    show_time(), epoch, batch_id, optimizer.param_groups[0]['lr'], float(loss), 
                    num_correct / num_tests, num_correct_packet / num_tests_packet))
        
        # 每个epoch结束后打印平均损失和准确率
        avg_loss = sum(loss_all) / len(loss_all)
        avg_acc = num_correct / num_tests
        avg_packet_acc = num_correct_packet / num_tests_packet
        
        print('{} Epoch {} finished, avg_loss: {:.4f}, avg_acc: {:.3f}, avg_packet_acc: {:.3f}'.format(
            show_time(), epoch, avg_loss, avg_acc, avg_packet_acc))

    # 保存微调后的模型
    torch.save(model.state_dict(), config.MIX_MODEL_CHECKPOINT[:-4] + '_finetuned_' + str(opt.prefix) + '.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="dataset", required=True)
    parser.add_argument("--num_workers", type=int, help="num workers", default=-1)
    parser.add_argument("--prefix", type=str, required=True)
    parser.add_argument("--load_prefix", type=str, required=True, help="prefix of pretrained model")
    parser.add_argument("--point", type=int, default=15)
    parser.add_argument("--seq_aug_ratio", type=float, default=0.8)
    parser.add_argument("--tau", type=float, default=0.07)
    parser.add_argument("--gtau", type=float, default=0.07)
    parser.add_argument("--drop_edge_ratio", type=float, default=0.1)
    parser.add_argument("--drop_node_ratio", type=float, default=0.1)
    parser.add_argument("--K", type=int, default=15)
    parser.add_argument("--hp_ratio", type=float, default=0.5)
    parser.add_argument("--perc", type=float, default=1.0)
    parser.add_argument("--bs", type=int, default=-1)
    parser.add_argument("--ga", type=int, default=-1)
    parser.add_argument("--freeze_encoder", action="store_true", help="whether to freeze encoder during fine-tuning")
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

    # 微调配置
    config.FINETUNE_EPOCH = 30  # 可调整
    config.FINETUNE_LR = config.LR * 0.1  # 通常微调使用更小的学习率

    device = get_device(index=0)
    num_workers =0
    set_seed()
    finetune()