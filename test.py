# test.py
import argparse
import torch
import torch.nn as nn
from dgl.dataloading import GraphDataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from dataloader import MixTrafficFlowDataset4DGL
from model_new_aug import MixTemporalGNN_SSL
from utils import show_time, set_seed, get_device, mix_collate_cl_fn
from config import *

def test():
    # 创建模型实例
    model = MixTemporalGNN_SSL(num_classes=config.NUM_CLASSES, embedding_size=config.EMBEDDING_SIZE, h_feats=config.H_FEATS,
                           dropout=0.0, downstream_dropout=0.0, point=opt.point,  # 测试时不需要dropout
                           seq_aug_ratio=opt.seq_aug_ratio, drop_edge_ratio=opt.drop_edge_ratio,
                           drop_node_ratio=opt.drop_node_ratio, K=opt.K, hp_ratio=opt.hp_ratio, tau=opt.tau, gtau=opt.gtau)
    
    # 加载微调后的权重
    model_path = config.MIX_MODEL_CHECKPOINT[:-4] + '_finetuned_' + str(opt.model_prefix) + '.pth'
    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path))
    
    # 加载测试数据
    test_dataset = MixTrafficFlowDataset4DGL(header_path=config.HEADER_TEST_GRAPH_DATA,
                                           payload_path=config.TEST_GRAPH_DATA,
                                           point=opt.point,
                                           perc=opt.perc)
    
    test_dataloader = GraphDataLoader(test_dataset, batch_size=config.BATCH_SIZE if opt.bs == -1 else opt.bs, 
                                    shuffle=False, collate_fn=mix_collate_cl_fn,
                                    num_workers=num_workers, pin_memory=True)
    
    model = model.to(device)
    model.eval()  # 设置为评估模式
    
    # 用于存储预测结果
    all_preds = []
    all_labels = []
    all_packet_preds = []
    all_packet_labels = []
    
    with torch.no_grad():  # 不计算梯度
        for batch_id, (header_data, payload_data, labels, header_mask, payload_mask) in enumerate(test_dataloader):
            header_data = header_data.to(device, non_blocking=True)
            payload_data = payload_data.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            #header_mask = header_mask.to(device, non_blocking=True)
            #payload_mask = payload_mask.to(device, non_blocking=True)
            
            # 前向传播，只使用分类部分
            pred, packet_out, packet_label, _ = model.forward_finetune(
                header_data, payload_data, labels, header_mask, payload_mask)
            
            # 收集预测结果
            all_preds.extend(pred.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_packet_preds.extend(packet_out.argmax(1).cpu().numpy())
            all_packet_labels.extend(packet_label.cpu().numpy())
            
            if batch_id % 10 == 0:
                print(f"{show_time()} Testing batch {batch_id}/{len(test_dataloader)}")
    
    # 计算流级别性能指标
    flow_accuracy = accuracy_score(all_labels, all_preds)
    flow_precision, flow_recall, flow_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted')
    flow_cm = confusion_matrix(all_labels, all_preds)
    
    # 计算包级别性能指标
    packet_accuracy = accuracy_score(all_packet_labels, all_packet_preds)
    packet_precision, packet_recall, packet_f1, _ = precision_recall_fscore_support(
        all_packet_labels, all_packet_preds, average='weighted')
    packet_cm = confusion_matrix(all_packet_labels, all_packet_preds)
    
    # 打印结果
    print("\n" + "="*50)
    print(f"Test Results for {opt.model_prefix}")
    print("="*50)
    
    print("\nFlow-level Metrics:")
    print(f"Accuracy: {flow_accuracy:.4f}")
    print(f"Precision: {flow_precision:.4f}")
    print(f"Recall: {flow_recall:.4f}")
    print(f"F1 Score: {flow_f1:.4f}")
    print("\nConfusion Matrix:")
    print(flow_cm)
    
    print("\nPacket-level Metrics:")
    print(f"Accuracy: {packet_accuracy:.4f}")
    print(f"Precision: {packet_precision:.4f}")
    print(f"Recall: {packet_recall:.4f}")
    print(f"F1 Score: {packet_f1:.4f}")
    print("\nConfusion Matrix:")
    print(packet_cm)
    
    # 保存结果到文件
    with open(f"test_results_{opt.model_prefix}.txt", "w") as f:
        f.write(f"Test Results for {opt.model_prefix}\n")
        f.write("="*50 + "\n")
        
        f.write("\nFlow-level Metrics:\n")
        f.write(f"Accuracy: {flow_accuracy:.4f}\n")
        f.write(f"Precision: {flow_precision:.4f}\n")
        f.write(f"Recall: {flow_recall:.4f}\n")
        f.write(f"F1 Score: {flow_f1:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(str(flow_cm) + "\n")
        
        f.write("\nPacket-level Metrics:\n")
        f.write(f"Accuracy: {packet_accuracy:.4f}\n")
        f.write(f"Precision: {packet_precision:.4f}\n")
        f.write(f"Recall: {packet_recall:.4f}\n")
        f.write(f"F1 Score: {packet_f1:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(str(packet_cm) + "\n")
    
    print(f"\nResults saved to test_results_{opt.model_prefix}.txt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="dataset", required=True)
    parser.add_argument("--num_workers", type=int, help="num workers", default=-1)
    parser.add_argument("--model_prefix", type=str, required=True, help="prefix of finetuned model")
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

    device = get_device(index=0)
    num_workers = 0
    set_seed()
    test()