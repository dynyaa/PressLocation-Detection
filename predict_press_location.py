import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import joblib
import numpy as np
import torchvision.models as models

import cv2

def extract_key_frame(video_path, frame_number):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in the video: {total_frames}")

    if frame_number >= total_frames:
        print(f"Error: Frame number {frame_number} exceeds total frames {total_frames}.")
        return None

    cap.set(1, frame_number) 
    ret, frame = cap.read()
    if ret:
        frame_path = 'key_frame.jpg'
        cv2.imwrite(frame_path, frame)
        return frame_path
    else:
        print("Failed to extract frame")
        return None


def preprocess_frame(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0) 
    return image

def extract_features(image_tensor, model):
    with torch.no_grad():
        features = model(image_tensor)
    return features.squeeze().numpy()

cnn_model = models.resnet18(pretrained=True)
cnn_model = torch.nn.Sequential(*list(cnn_model.children())[:-1])  
cnn_model.eval()

classifier = joblib.load('logistic_regression_model.pkl')

def predict_location(image_path):
    image_tensor = preprocess_frame(image_path)
    
    
    features = extract_features(image_tensor, cnn_model)
    

    features = features.reshape(1, -1)
    

    prediction = classifier.predict(features)
    
    return prediction[0]

video_path = r"C:\Users\robot\Downloads\dataset\touch\videos\bottom(video)\Flipped_WIN_20240702_17_55_02_Pro.mp4"
frame_number = 10  #
key_frame_path = extract_key_frame(video_path, frame_number)

if key_frame_path:
    predicted_location = predict_location(key_frame_path)
    print(f'The pressing location is predicted to be: {predicted_location}')

