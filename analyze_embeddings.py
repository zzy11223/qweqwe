# analyze_embeddings.py
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from dgl.dataloading import GraphDataLoader

from dataloader import MixTrafficFlowDataset4DGL
from model_new_aug import MixTemporalGNN_SSL
from utils import show_time, set_seed, get_device, mix_collate_cl_fn
from config import *

def analyze_embeddings():
    # 创建模型实例
    model = MixTemporalGNN_SSL(num_classes=config.NUM_CLASSES, embedding_size=config.EMBEDDING_SIZE, h_feats=config.H_FEATS,
                           dropout=0.0, downstream_dropout=0.0, point=opt.point,
                           seq_aug_ratio=opt.seq_aug_ratio, drop_edge_ratio=opt.drop_edge_ratio,
                           drop_node_ratio=opt.drop_node_ratio, K=opt.K, hp_ratio=opt.hp_ratio, tau=opt.tau, gtau=opt.gtau)
    
    # 加载模型权重
    if opt.model_type == 'pretrained':
        model_path = config.MIX_MODEL_CHECKPOINT[:-4] + '_pretrained_' + str(opt.model_prefix) + '.pth'
    elif opt.model_type == 'finetuned':
        model_path = config.MIX_MODEL_CHECKPOINT[:-4] + '_finetuned_' + str(opt.model_prefix) + '.pth'
    else:
        raise ValueError("Model type must be 'pretrained' or 'finetuned'")
    
    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path))
    
    # 加载数据
    dataset = MixTrafficFlowDataset4DGL(header_path=config.HEADER_TEST_GRAPH_DATA,
                                      payload_path=config.TEST_GRAPH_DATA,
                                      point=opt.point,
                                      perc=opt.perc)
    
    dataloader = GraphDataLoader(dataset, batch_size=config.BATCH_SIZE if opt.bs == -1 else opt.bs, 
                               shuffle=False, collate_fn=mix_collate_cl_fn,
                               num_workers=num_workers, pin_memory=True)
    
    model = model.to(device)
    model.eval()
    
    # 收集表示和标签
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for batch_id, (header_data, payload_data, labels, header_mask, payload_mask) in enumerate(dataloader):
            header_data = header_data.to(device, non_blocking=True)
            payload_data = payload_data.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            
            # 获取表示
            if opt.model_type == 'pretrained':
                # 对于预训练模型，获取对比学习表示
                representations, _, _ = model.forward_pretrain(header_data, payload_data, header_mask, payload_mask)
            else:
                # 对于微调模型，获取最终表示
                _, _, _, representations = model.forward_finetune(header_data, payload_data, labels, header_mask, payload_mask)
            
            all_embeddings.append(representations.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            if batch_id % 10 == 0:
                print(f"{show_time()} Processing batch {batch_id}/{len(dataloader)}")
    
    # 合并所有批次的数据
    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    print(f"Collected {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
    
    # 降维到2D以便可视化
    print("Applying dimensionality reduction...")
    if opt.viz_method == 'tsne':
        # 使用t-SNE进行降维
        embeddings_2d = TSNE(n_components=2, random_state=42).fit_transform(embeddings)
        method_name = 't-SNE'
    elif opt.viz_method == 'pca':
        # 使用PCA进行降维
        embeddings_2d = PCA(n_components=2, random_state=42).fit_transform(embeddings)
        method_name = 'PCA'
    else:
        raise ValueError("Visualization method must be 'tsne' or 'pca'")
    
    # 可视化
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                  c=[colors[i]], label=f'Class {label}', alpha=0.7)
    
    plt.title(f'{method_name} visualization of {opt.model_type} embeddings')
    plt.legend()
    plt.tight_layout()
    
    # 保存图像
    filename = f'{opt.model_type}_{opt.model_prefix}_{opt.viz_method}.png'
    plt.savefig(filename, dpi=300)
    print(f"Visualization saved as {filename}")
    
    # 计算类间类内距离
    print("\nCalculating distances between embeddings...")
    
    # 计算每个类的中心点
    class_centers = {}
    for label in unique_labels:
        mask = labels == label
        class_centers[label] = np.mean(embeddings[mask], axis=0)
    
    # 计算类内距离
    intra_class_distances = []
    for label in unique_labels:
        mask = labels == label
        class_embeddings = embeddings[mask]
        center = class_centers[label]
        distances = np.linalg.norm(class_embeddings - center, axis=1)
        intra_class_distances.append(np.mean(distances))
    
    avg_intra_class_distance = np.mean(intra_class_distances)
    
    # 计算类间距离
    inter_class_distances = []
    for i in range(len(unique_labels)):
        for j in range(i+1, len(unique_labels)):
            dist = np.linalg.norm(class_centers[unique_labels[i]] - class_centers[unique_labels[j]])
            inter_class_distances.append(dist)
    
    avg_inter_class_distance = np.mean(inter_class_distances)
    
    # 输出结果
    print(f"Average intra-class distance: {avg_intra_class_distance:.4f}")
    print(f"Average inter-class distance: {avg_inter_class_distance:.4f}")
    print(f"Ratio (inter/intra): {avg_inter_class_distance/avg_intra_class_distance:.4f}")
    
    # 保存结果到文件
    with open(f"embedding_analysis_{opt.model_type}_{opt.model_prefix}.txt", "w") as f:
        f.write(f"Embedding Analysis for {opt.model_type} model {opt.model_prefix}\n")
        f.write("="*50 + "\n")
        f.write(f"Number of embeddings: {len(embeddings)}\n")
        f.write(f"Embedding dimension: {embeddings.shape[1]}\n\n")
        f.write(f"Average intra-class distance: {avg_intra_class_distance:.4f}\n")
        f.write(f"Average inter-class distance: {avg_inter_class_distance:.4f}\n")
        f.write(f"Ratio (inter/intra): {avg_inter_class_distance/avg_intra_class_distance:.4f}\n")
        
        f.write("\nPer-class intra-class distances:\n")
        for i, label in enumerate(unique_labels):
            f.write(f"Class {label}: {intra_class_distances[i]:.4f}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="dataset", required=True)
    parser.add_argument("--num_workers", type=int, help="num workers", default=-1)
    parser.add_argument("--model_prefix", type=str, required=True, help="prefix of model")
    parser.add_argument("--model_type", type=str, required=True, choices=['pretrained', 'finetuned'], 
                       help="model type (pretrained or finetuned)")
    parser.add_argument("--viz_method", type=str, default='tsne', choices=['tsne', 'pca'],
                       help="visualization method (tsne or pca)")
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
    analyze_embeddings()