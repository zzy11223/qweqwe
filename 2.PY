import os
import shutil

def move_files_with_tcp(source_folder, destination_folder):
    # 检查源文件夹是否存在
    if not os.path.exists(source_folder):
        print(f"源文件夹 {source_folder} 不存在。")
        return

    # 检查目标文件夹是否存在，如果不存在则创建
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        print(f"目标文件夹 {destination_folder} 已创建。")

    # 遍历源文件夹中的所有文件
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if "tcp" in file.lower():
                source_file_path = os.path.join(root, file)
                destination_file_path = os.path.join(destination_folder, file)
                try:
                    # 移动文件
                    shutil.move(source_file_path, destination_file_path)
                    print(f"已将文件 {source_file_path} 移动到 {destination_file_path}")
                except Exception as e:
                    print(f"移动文件 {source_file_path} 时出错: {e}")

if __name__ == "__main__":
    # 指定源文件夹和目标文件夹的路径
    source_folder = r'E:\GITHUB code\Dataset\ process'
    destination_folder = "./Datasets/NonTor_tcp"

    move_files_with_tcp(source_folder, destination_folder)