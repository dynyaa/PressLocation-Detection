import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd

model = models.resnet18(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove the final classification layer
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

split_data_path = 'split_data'

def extract_features(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  
    with torch.no_grad():
        features = model(image)
    return features.squeeze().numpy()

for set_name in ['train', 'val', 'test']:
    set_path = os.path.join(split_data_path, set_name)
    features = []
    labels = []
    for label in os.listdir(set_path):
        label_path = os.path.join(set_path, label)
        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            feature = extract_features(img_path)
            features.append(feature)
            labels.append(label)
    
    features_df = pd.DataFrame(features)
    features_df['label'] = labels
    features_df.to_csv(f'{set_name}_features.csv', index=False)
