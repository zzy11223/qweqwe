import os
import shutil

def filter_files(txt_folder, source_folder, output_folder):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    all_required_files = set()
    for txt_file in os.listdir(txt_folder):
        if txt_file.endswith('.txt'):
            txt_path = os.path.join(txt_folder, txt_file)
            with open(txt_path, 'r', encoding='utf-8') as f:
                all_required_files.update(line.strip() for line in f)
    
    all_actual_files = set(os.listdir(source_folder))
    
    # 找出缺少和多出的文件
    missing_files = all_required_files - all_actual_files
    extra_files = all_actual_files - all_required_files
    
    # 复制符合要求的文件到目标文件夹
    for file in all_required_files.intersection(all_actual_files):
        shutil.copy(os.path.join(source_folder, file), os.path.join(output_folder, file))
    
    # 保存多出的和缺少的文件列表
    with open(os.path.join(output_folder, 'missing_files.txt'), 'w', encoding='utf-8') as f:
        f.writelines(f"{file}\n" for file in sorted(missing_files))
    
    with open(os.path.join(output_folder, 'extra_files.txt'), 'w', encoding='utf-8') as f:
        f.writelines(f"{file}\n" for file in sorted(extra_files))
    
    print("处理完成，结果保存在输出目录。")



# 示例调用
filter_files('.\CATE\ISCX-NonTor', '.\Datasets/NonTor_tcp', '.\Datasets/1')
#txt      ，source    ，destination

