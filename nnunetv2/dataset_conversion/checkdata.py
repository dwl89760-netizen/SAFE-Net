pip install nibabel
import nibabel as nib
import numpy as np

# 读取标签数据
label_path = 'your_label_path.nii.gz'
label_obj = nib.load(label_path)
label_data = label_obj.get_fdata()

# 获取标签名称
label_header = label_obj.header
print(label_header)
# 查看标签中的唯一值
unique_labels = np.unique(label_data)
print(f'Unique labels: {unique_labels}')

# 查看每个标签对应的像素数量
unique_labels, counts = np.unique(label_data, return_counts=True)
print(f'Labels and their pixel counts: {dict(zip(unique_labels, counts))}')

# 查看标签名称
if 'descrip' in label_header:
    print(f"Label description: {label_header['descrip']}")
else:
    print("No label description found in the header.")
