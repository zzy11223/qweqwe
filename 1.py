import os
import shutil

def sort_files_from_multiple_sources(source_folders, target_folder):
    # 确保目标文件夹及分类子文件夹存在
    categories = ['chat', 'email', 'ftp', 'audio', 'p2p', 'voip', 'browsing', 'video']
    for category in categories:
        category_path = os.path.join(target_folder, category)
        os.makedirs(category_path, exist_ok=True)
    
    # 遍历所有源文件夹
    for folder_path in source_folders:
        if not os.path.isdir(folder_path):
            print(f"警告: {folder_path} 不是有效的文件夹，跳过。")
            continue
        
        # 遍历文件夹中的文件
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            
            # 只处理文件
            if os.path.isfile(file_path):
                for category in categories:
                    if category in file_name.lower():  # 忽略大小写匹配
                        target_path = os.path.join(target_folder, category, file_name)
                        shutil.move(file_path, target_path)
                        print(f'Moved: {file_name} -> {category}/')
                        break  # 一个文件只属于一个类别

if __name__ == "__main__":
    source_folders = [
        
        r"E:\GITHUB code\TFE-GNN-main\Datasets\NonTor\NonTor\output_directory"
       
    ]
    target_folder = r"E:\GITHUB code\TFE-GNN-main\Datasets\NonTor"
    
    sort_files_from_multiple_sources(source_folders, target_folder)
    print("文件分类完成！")
