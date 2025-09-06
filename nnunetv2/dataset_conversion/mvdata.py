import os
import shutil

def process_crop_data(source_dir, target_dir):
    # 创建目标目录及其子目录
    imagesTr_dir = os.path.join(target_dir, "imagesTr")
    imagesTs_dir = os.path.join(target_dir, "imagesTs")
    labelsTr_dir = os.path.join(target_dir, "labelsTr")
    os.makedirs(imagesTr_dir, exist_ok=True)
    os.makedirs(imagesTs_dir, exist_ok=True)
    os.makedirs(labelsTr_dir, exist_ok=True)

    # 初始化计数器
    counter = 1

    # 获取crop_data目录下的所有子文件夹
    subfolders = [f.path for f in os.scandir(source_dir) if f.is_dir()]

    for subfolder in subfolders:
        # 遍历每个子文件夹，查找名字包含.nii.gz和_gt.nii.gz的文件
        for file in os.listdir(subfolder):
            source_path = os.path.join(subfolder, file)
            if file.endswith(".nii.gz") and not file.endswith("_gt.nii.gz"):
                # 复制到imagesTr目录
                new_filename = f"{os.path.basename(subfolder)}_{str(counter).zfill(3)}.nii.gz"
                shutil.copy(source_path, os.path.join(imagesTr_dir, new_filename))
                print(f"Copied {file} to {new_filename} in imagesTr")
            elif file.endswith("_gt.nii.gz"):
                # 复制到labelsTr目录
                new_filename = f"{os.path.basename(subfolder)}_{str(counter).zfill(3)}.nii.gz"
                shutil.copy(source_path, os.path.join(labelsTr_dir, new_filename))
                print(f"Copied {file} to {new_filename} in labelsTr")
        counter += 1

# 指定源目录和目标目录
source_directory = r"/data/yan/nnUNet-master/crop_data"
target_directory = r"/data/yan/nnUNet-master/Task508_CTA"

# 执行处理操作
process_crop_data(source_directory, target_directory)
