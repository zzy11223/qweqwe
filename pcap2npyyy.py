import argparse
import numpy as np
import subprocess
import os
from scapy.all import *
from scapy.utils import PcapReader
import re
from utils import show_time, get_bytes_from_raw
from config import *


class PCAPProcessor:
    def __init__(self, max_packets=10000):
        self.max_packets = max_packets
        self.reset()
        
    def reset(self):
        self.PKT_COUNT = 0
        self.p_header_list = []
        self.p_payload_list = []
        self.payload_length = []
        self.pkt_length = []
        self.src_ip = []
        self.dst_ip = []
        self.src_port = []
        self.dst_port = []
        self.time = []
        self.protocol = []
        self.flag = []
        self.mss = []
        self.has_payload = False

    
    def _get_pcap_count(self, file_path):

        try:
            result = subprocess.run(
                ['capinfos', '-c', file_path],
                capture_output=True,
                text=True,
                check=True
            )
            # 更健壮的解析方式
            for line in result.stdout.split('\n'):
                if 'Number of packets' in line:
                    # 使用正则表达式提取数字
                    match = re.search(r'\d+', line)
                    if match:
                        return int(match.group())
                    else:
                        print(f"{show_time()} 无法解析数据包数量: {line.strip()}")
                        return None
            return None
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"{show_time()} capinfos错误: {str(e)}")
            return self._fallback_pcap_count(file_path)
        except Exception as e:
            print(f"{show_time()} 未知错误: {str(e)}")
            return None







    def _fallback_pcap_count(self, file_path):
        """回退方案：快速扫描计数"""
        print(f"{show_time()} 使用回退方案统计 {file_path}")
        count = 0
        try:
            with PcapReader(file_path) as pcap_reader:
                while True:
                    pkt = pcap_reader.read_packet()
                    if pkt is None:
                        break
                    count += 1
                    if count > self.max_packets:  # 提前终止
                        break
        except Exception as e:
            print(f"{show_time()} 回退方案错误: {str(e)}")
        return count

    def _packet_handler(self, pkt):
        """数据包处理核心逻辑"""
        if self.PKT_COUNT >= self.max_packets:
            return
        
        self.PKT_COUNT += 1
        
        # 解析原始字节
        _, p_packet = get_bytes_from_raw(hexdump(pkt, dump=True))
        
        # 处理有效载荷
        p_payload = []
        if pkt.haslayer("Raw"):
            _, p_payload = get_bytes_from_raw(hexdump(pkt["Raw"].load, dump=True))
            if len(p_payload) > 0:
                self.has_payload = True

        # 存储数据
        p_header = p_packet[:(len(p_packet) - len(p_payload))]
        self.p_header_list.append(p_header)
        self.p_payload_list.append(p_payload)
        self.payload_length.append(len(p_payload))
        self.pkt_length.append(len(p_header) + len(p_payload))
        
        # 元数据提取
        self._extract_metadata(pkt)

    def _extract_metadata(self, pkt):
        """提取网络元数据"""
        self.src_ip.append(pkt.src)
        self.dst_ip.append(pkt.dst)
        self.src_port.append(pkt.sport)
        self.dst_port.append(pkt.dport)
        self.time.append(pkt.time)
        self.protocol.append(pkt.proto)
        
        # TCP特定字段
        tcp_flags = 0
        tcp_mss = 0
        if pkt.haslayer('TCP'):
            tcp_flags = pkt['TCP'].flags
            for opt, val in pkt['TCP'].options:
                if opt == 'MSS':
                    tcp_mss = val
                    break
        self.flag.append(tcp_flags)
        self.mss.append(tcp_mss)

    def process_file(self, input_path, output_path):
        """处理单个pcap文件"""
        self.reset()
        
        # 预扫描阶段
        pcap_count = self._get_pcap_count(input_path)
        if pcap_count is None:
            print(f"{show_time()} 无法获取数据包数量: {input_path}")
            return False

        # 跳过条件判断
        if pcap_count == 0:
            print(f"{show_time()} 跳过空文件: {input_path}")
            return False
        if pcap_count > self.max_packets:
            print(f"{show_time()} 跳过过长流: {input_path} ({pcap_count} packets)")
            return False

        # 正式处理阶段
        print(f"{show_time()} 开始处理: {input_path}")
        try:
            sniff(offline=input_path, prn=self._packet_handler, store=0)
        except Scapy_Exception as e:
            print(f"{show_time()} 处理错误: {str(e)}")
            return False

        # 后处理检查
        if not self.has_payload:
            print(f"{show_time()} 跳过空有效载荷流: {input_path}")
            return False
        if self.PKT_COUNT == 0:
            print(f"{show_time()} 无有效数据包: {input_path}")
            return False

        # 保存结果
        self._save_results(output_path)
        return True

    def _save_results(self, output_path):
        """保存处理结果到npz文件"""
        np.savez_compressed(
            output_path,
            header=np.array(self.p_header_list, dtype=object),
            payload=np.array(self.p_payload_list, dtype=object),
            payload_length=np.array(self.payload_length, dtype=int),
            pkt_length=np.array(self.pkt_length, dtype=int),
            src_ip=np.array(self.src_ip, dtype=object),
            dst_ip=np.array(self.dst_ip, dtype=object),
            src_port=np.array(self.src_port, dtype=int),
            dst_port=np.array(self.dst_port, dtype=int),
            time=np.array(self.time, dtype=float),
            protocol=np.array(self.protocol, dtype=int),
            flag=np.array(self.flag, dtype=int),
            mss=np.array(self.mss, dtype=int)
        )
        print(f"{show_time()} 保存完成: {output_path}")


def pcap2npy4ISCX(dir_path_dict, save_path_dict, max_packets=10000):
    processor = PCAPProcessor(max_packets=max_packets)
    
    for category in dir_path_dict:
        input_dir = dir_path_dict[category]
        output_dir = save_path_dict[category]
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        for filename in os.listdir(input_dir):
            if not filename.endswith('.pcap'):
                continue
                
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(
                output_dir,
                filename.replace('.pcap', '.npz')
            )
            
            if os.path.exists(output_path):
                print(f"{show_time()} 跳过已处理文件: {filename}")
                continue
                
            processor.process_file(input_path, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                       choices=['iscx-vpn', 'iscx-nonvpn', 'iscx-tor', 'iscx-nontor'],
                       help="数据集名称")
    parser.add_argument("--max_packets", type=int, default=10000,
                       help="最大处理数据包数")
    args = parser.parse_args()

    # 配置加载
    config_map = {
        'iscx-vpn': ISCXVPNConfig(),
        'iscx-nonvpn': ISCXNonVPNConfig(),
        'iscx-tor': ISCXTorConfig(),
        'iscx-nontor': ISCXNonTorConfig()
    }
    
    try:
        config = config_map[args.dataset]
        pcap2npy4ISCX(
            dir_path_dict=config.DIR_PATH_DICT,
            save_path_dict=config.DIR_PATH_DICT,
            max_packets=args.max_packets
        )
    except KeyError:
        raise ValueError(f"不支持的dataset参数: {args.dataset}")