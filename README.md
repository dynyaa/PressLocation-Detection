# PressLocation-Detection
Task:
The objective of this project was to develop a machine learning pipeline that classifies the location of pressing on a silicone surface, based on video data. The videos capture different pressing locations, and the goal is to accurately identify these locations using a trained model.

Approach:
Data Preparation:

Frame Extraction and Labeling: Extracted key frames from labeled videos that depict pressing actions at different locations on the silicone surface.
Data Splitting: Organized the extracted frames into training, validation, and test sets to ensure robust model evaluation.
Feature Extraction:

Utilized a pre-trained ResNet-18 Convolutional Neural Network (CNN) to extract deep features from the frames. The final classification layer of ResNet-18 was removed, enabling the network to serve purely as a feature extractor.
Model Training:

Trained a Logistic Regression model on the extracted features to classify the pressing locations.
The model was validated on a separate validation set and tested on unseen data, achieving high accuracy.
Model Deployment:

Developed a pipeline that allows for the prediction of pressing locations on new video data. The pipeline extracts frames from videos, preprocesses them, extracts features using ResNet-18, and classifies the location using the trained Logistic Regression model.
