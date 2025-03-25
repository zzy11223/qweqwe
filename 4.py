import os
import shutil

def classify_files(source_folder, txt_folder, destination_folder):
    # 检查源文件夹、txt文件夹和目标文件夹是否存在
    if not os.path.exists(source_folder):
        print(f"源文件夹 {source_folder} 不存在。")
        return
    if not os.path.exists(txt_folder):
        print(f"txt文件夹 {txt_folder} 不存在。")
        return
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # 遍历txt文件夹中的所有txt文件
    for txt_file in os.listdir(txt_folder):
        if txt_file.endswith('.txt'):
            # 获取当前txt文件的名称（不包含扩展名）
            category_name = os.path.splitext(txt_file)[0]
            # 创建对应的子文件夹
            category_folder = os.path.join(destination_folder, category_name)
            if not os.path.exists(category_folder):
                os.makedirs(category_folder)

            # 读取txt文件中的文件名
            with open(os.path.join(txt_folder, txt_file), 'r', encoding='utf-8') as f:
                file_names = [line.strip() for line in f.readlines()]

            # 遍历源文件夹中的所有文件
            for file in os.listdir(source_folder):
                if file in file_names:
                    # 移动文件到对应的子文件夹
                    source_file_path = os.path.join(source_folder, file)
                    destination_file_path = os.path.join(category_folder, file)
                    shutil.move(source_file_path, destination_file_path)
                    print(f"已将 {file} 移动到 {category_folder}")

if __name__ == "__main__":
    # 指定源文件夹，包含需要分类的文件
    source_folder = '.\Datasets/NONVPN_TCP'
    # 指定包含txt文件的文件夹
    txt_folder = '.\CATE\ISCX-NonVPN'
    # 指定目标文件夹，用于存放分类后的文件
    destination_folder = '.\Datasets/NonVPNtcpdown'

    classify_files(source_folder, txt_folder, destination_folder)


