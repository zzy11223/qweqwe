'''
Training Configuration
'''
class Config:
    BATCH_SIZE = 102
    GRADIENT_ACCUMULATION = 5
    MAX_EPOCH = 20
    LR = 1e-2
    LR_MIN = 1e-4
    LABEL_SMOOTHING = 0
    WEIGHT_DECAY = 0
    WARM_UP = 0.1
    SEED = 32
    DROPOUT = 0.2
    DOWNSTREAM_DROPOUT = 0.0

    EMBEDDING_SIZE = 64
    H_FEATS = 128
    NUM_CLASSES = 14

    PMI_WINDOW_SIZE = 5
    PAD_TRUNC_DIGIT = 256
    FLOW_PAD_TRUNC_LENGTH = 50
    BYTE_PAD_TRUNC_LENGTH = 150
    HEADER_BYTE_PAD_TRUNC_LENGTH = 50
    ANOMALOUS_FLOW_THRESHOLD = 10000

'''
ISCX-VPN Dataset Configuration
'''
class ISCXVPNConfig(Config):
    TRAIN_DATA = r'E:\GITHUB code\TFE-GNN-main\Datasets\VPNtcpdown\TCP/train.npz'
    HEADER_TRAIN_DATA = r'E:\GITHUB code\TFE-GNN-main\Datasets\VPNtcpdown\TCP/header_train.npz'
    TEST_DATA = r'E:\GITHUB code\TFE-GNN-main\Datasets\VPNtcpdown\TCP/test.npz'
    HEADER_TEST_DATA = r'E:\GITHUB code\TFE-GNN-main\Datasets\VPNtcpdown\TCP/header_test.npz'

    TRAIN_GRAPH_DATA = r'E:\GITHUB code\TFE-GNN-main\Datasets\VPNtcpdown\TCP/train_graph.dgl'
    HEADER_TRAIN_GRAPH_DATA = r'E:\GITHUB code\TFE-GNN-main\Datasets\VPNtcpdown\TCP/header_train_graph.dgl'
    TEST_GRAPH_DATA = r'E:\GITHUB code\TFE-GNN-main\Datasets\VPNtcpdown\TCP/test_graph.dgl'
    HEADER_TEST_GRAPH_DATA = r'E:\GITHUB code\TFE-GNN-main\Datasets\VPNtcpdown\TCP/header_test_graph.dgl'

    MIX_MODEL_CHECKPOINT = r'./checkpoints/mix_model_iscx_vpn.pth'

    NUM_CLASSES = 6
    MAX_SEG_PER_CLASS = 9999
    NUM_WORKERS = 5

    BATCH_SIZE = 64

    GRADIENT_ACCUMULATION = 1
    MAX_EPOCH = 20
    LR = 1e-2
    LR_MIN = 1e-4
    LABEL_SMOOTHING = 0
    WEIGHT_DECAY = 0
    WARM_UP = 0.1
    SEED = 32
    DROPOUT = 0.0
    DOWNSTREAM_DROPOUT = 0.0
    EMBEDDING_SIZE = 64
    H_FEATS = 128
    

    DIR_PATH_DICT = {0: r'E:\GITHUB code\TFE-GNN-main\Datasets\VPNtcpdown/Chat',
                     1: r'E:\GITHUB code\TFE-GNN-main\Datasets\VPNtcpdown/Email',
                     2: r'E:\GITHUB code\TFE-GNN-main\Datasets\VPNtcpdown/File',
                     3: r'E:\GITHUB code\TFE-GNN-main\Datasets\VPNtcpdown/P2P',
                     4: r'E:\GITHUB code\TFE-GNN-main\Datasets\VPNtcpdown/Streaming',
                     5: r'E:\GITHUB code\TFE-GNN-main\Datasets\VPNtcpdown/VoIP',
                     }


'''
ISCX-NonVPN Dataset Configuration
'''
class ISCXNonVPNConfig(Config):
    TRAIN_DATA = r'E:/GITHUB code\TFE-GNN-main\Datasets\NonVPNtcpdown\tcp/train.npz'
    HEADER_TRAIN_DATA = r'E:/GITHUB code\TFE-GNN-main\Datasets\NonVPNtcpdown\tcp/header_train.npz'
    TEST_DATA = r'E:/GITHUB code\TFE-GNN-main\Datasets\NonVPNtcpdown\tcp/test.npz'
    HEADER_TEST_DATA = r'E:/GITHUB code\TFE-GNN-main\Datasets\NonVPNtcpdown\tcp/header_test.npz'

    TRAIN_GRAPH_DATA = r'E:/GITHUB code\TFE-GNN-main\Datasets\NonVPNtcpdown\tcp/train_graph.dgl'
    HEADER_TRAIN_GRAPH_DATA = r'E:/GITHUB code\TFE-GNN-main\Datasets\NonVPNtcpdown\tcp/header_train_graph.dgl'
    TEST_GRAPH_DATA = r'E:/GITHUB code\TFE-GNN-main\Datasets\NonVPNtcpdown\tcp/test_graph.dgl'
    HEADER_TEST_GRAPH_DATA = r'E:/GITHUB code\TFE-GNN-main\Datasets\NonVPNtcpdown\tcp/header_test_graph.dgl'

    MIX_MODEL_CHECKPOINT = r'./checkpoints/mix_model_iscx_nonvpn.pth'

    NUM_CLASSES = 6
    MAX_SEG_PER_CLASS = 9999
    NUM_WORKERS = 0

    BATCH_SIZE = 64
    GRADIENT_ACCUMULATION = 5
    MAX_EPOCH = 120
    LR = 1e-2
    LR_MIN = 1e-5
    LABEL_SMOOTHING = 0.01
    WEIGHT_DECAY = 0
    WARM_UP = 0.1
    SEED = 32
    DROPOUT = 0.1
    DOWNSTREAM_DROPOUT = 0.15
    EMBEDDING_SIZE = 64
    H_FEATS = 128

    DIR_PATH_DICT = {0: r'E:/GITHUB code\TFE-GNN-main\Datasets\NonVPNtcpdown/Chat',
                     1: r'E:/GITHUB code\TFE-GNN-main\Datasets\NonVPNtcpdown/Email',
                     2: r'E:/GITHUB code\TFE-GNN-main\Datasets\NonVPNtcpdown/File',
                     3: r'E:/GITHUB code\TFE-GNN-main\Datasets\NonVPNtcpdown/Streaming',
                     4: r'E:/GITHUB code\TFE-GNN-main\Datasets\NonVPNtcpdown/Video',
                     5: r'E:/GITHUB code\TFE-GNN-main\Datasets\NonVPNtcpdown/VoIP',
                     }


'''
ISCX-Tor Dataset Configuration
'''
class ISCXTorConfig(Config):
    TRAIN_DATA = r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/TOR_SPLIT/TCP/train.npz'
    HEADER_TRAIN_DATA = r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/TOR_SPLIT/TCP/header_train.npz'
    TEST_DATA = r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/TOR_SPLIT/TCP/test.npz'
    HEADER_TEST_DATA = r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/TOR_SPLIT/TCP/header_test.npz'

    TRAIN_GRAPH_DATA = r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/TOR_SPLIT/TCP/train_graph.dgl'
    HEADER_TRAIN_GRAPH_DATA = r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/TOR_SPLIT/TCP/header_train_graph.dgl'
    TEST_GRAPH_DATA = r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/TOR_SPLIT/TCP/test_graph.dgl'
    HEADER_TEST_GRAPH_DATA = r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/TOR_SPLIT/TCP/header_test_graph.dgl'

    MIX_MODEL_CHECKPOINT = r'./checkpoints/mix_model_iscx_tor.pth'

    NUM_CLASSES = 8
    MAX_SEG_PER_CLASS = 9999
    NUM_WORKERS = 5

    BATCH_SIZE = 32
    GRADIENT_ACCUMULATION = 1
    MAX_EPOCH = 100
    LR = 1e-2
    LR_MIN = 1e-4
    LABEL_SMOOTHING = 0
    WEIGHT_DECAY = 0
    WARM_UP = 0.1
    SEED = 32
    DROPOUT = 0.0
    DOWNSTREAM_DROPOUT = 0.0
    EMBEDDING_SIZE = 64
    H_FEATS = 128

    DIR_PATH_DICT = {0: r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/TOR_SPLIT/TCP/Audio-Streaming',
                     1: r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/TOR_SPLIT/TCP/Browsing',
                     2: r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/TOR_SPLIT/TCP/Chat',
                     3: r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/TOR_SPLIT/TCP/File',
                     4: r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/TOR_SPLIT/TCP/Mail',
                     5: r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/TOR_SPLIT/TCP/P2P',
                     6: r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/TOR_SPLIT/TCP/Video-Streaming',
                     7: r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/TOR_SPLIT/TCP/VoIP'
                     }


'''
ISCX-NonTor Dataset Configuration
'''
class ISCXNonTorConfig(Config):
    TRAIN_DATA = r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/NonTOR_SPLIT/TCP/train.npz'
    HEADER_TRAIN_DATA = r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/NonTOR_SPLIT/TCP/header_train.npz'
    TEST_DATA = r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/NonTOR_SPLIT/TCP/test.npz'
    HEADER_TEST_DATA = r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/NonTOR_SPLIT/TCP/header_test.npz'

    TRAIN_GRAPH_DATA = r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/NonTOR_SPLIT/TCP/train_graph.dgl'
    HEADER_TRAIN_GRAPH_DATA = r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/NonTOR_SPLIT/TCP/header_train_graph.dgl'
    TEST_GRAPH_DATA = r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/NonTOR_SPLIT/TCP/test_graph.dgl'
    HEADER_TEST_GRAPH_DATA = r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/NonTOR_SPLIT/TCP/header_test_graph.dgl'

    MIX_MODEL_CHECKPOINT = r'./checkpoints/mix_model_iscx_nontor.pth'

    NUM_CLASSES = 8
    MAX_SEG_PER_CLASS = 9999
    NUM_WORKERS = 5

    BATCH_SIZE = 102
    GRADIENT_ACCUMULATION = 5
    MAX_EPOCH = 120
    LR = 1e-2
    LR_MIN = 1e-4
    LABEL_SMOOTHING = 0
    WEIGHT_DECAY = 0
    WARM_UP = 0.1
    SEED = 32
    DROPOUT = 0.2
    DOWNSTREAM_DROPOUT = 0.1
    EMBEDDING_SIZE = 64
    H_FEATS = 128

    DIR_PATH_DICT = {0: r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/NonTOR_SPLIT/TCP/Audio',
                     1: r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/NonTOR_SPLIT/TCP/Browsing',
                     2: r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/NonTOR_SPLIT/TCP/Chat',
                     3: r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/NonTOR_SPLIT/TCP/Email',
                     4: r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/NonTOR_SPLIT/TCP/FTP',
                     5: r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/NonTOR_SPLIT/TCP/P2P',
                     6: r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/NonTOR_SPLIT/TCP/Video',
                     7: r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/NonTOR_SPLIT/TCP/VoIP',
                     }


if __name__ == '__main__':
    config = Config()
