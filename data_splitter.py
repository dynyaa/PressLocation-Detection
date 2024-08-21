import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil

data_path = r'C:\dataset(frames)\NUSense\touch'  


frames = []
labels = []


for label in os.listdir(data_path):
    folder_path = os.path.join(data_path, label)
    images = os.listdir(folder_path)
    for img in images:
        frames.append(os.path.join(folder_path, img))
        labels.append(label)

data = pd.DataFrame({
    'frames': frames,
    'labels': labels
})

print(data.head())

train_val_data, test_data = train_test_split(data, test_size=0.15, stratify=data['labels'], random_state=42)

train_data, val_data = train_test_split(train_val_data, test_size=0.18, stratify=train_val_data['labels'], random_state=42)

print(f"Training set size: {len(train_data)}")
print(f"Validation set size: {len(val_data)}")
print(f"Test set size: {len(test_data)}")

for set_name, dataset in [('train', train_data), ('val', val_data), ('test', test_data)]:
    for _, row in dataset.iterrows():
        label = row['labels']
        img_path = row['frames']
        dest_dir = os.path.join('split_data', set_name, label)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy(img_path, dest_dir)
