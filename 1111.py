import dgl
import torch

def analyze_dataset(graph_path):
    graphs, labels = dgl.load_graphs(graph_path)
    class_dist = torch.bincount(labels['glabel'])
    
    print(f"Total Graphs: {len(graphs)}")
    print("Class Distribution:")
    for cls, count in enumerate(class_dist):
        print(f"Class {cls}: {count} samples")
    
    avg_nodes = sum(g.num_nodes() for g in graphs) / len(graphs)
    avg_edges = sum(g.num_edges() for g in graphs) / len(graphs)
    print(f"\nAvg Nodes: {avg_nodes:.1f}, Avg Edges: {avg_edges:.1f}")
    # 输出标签的形状
    print(f"标签的形状: {labels['glabel'].shape}")

analyze_dataset(r'E:\GITHUB code\TFE-GNN-main\Datasets\VPN\TCP/train_graph.dgl')
# 示例代码查看图结构
graphs, labels = dgl.load_graphs(r'E:\GITHUB code\TFE-GNN-main\Datasets\VPN\TCP/train_graph.dgl')
sample_graph = graphs[0]

print(sample_graph)
# 查看节点特征
if len(sample_graph.ndata) > 0:
    print("节点特征:")
    for key in sample_graph.ndata:
        print(f"特征名: {key}, 特征值: {sample_graph.ndata[key]}")
else:
    print("图中没有节点特征。")