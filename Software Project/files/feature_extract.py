import os
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np

def extract_features(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    model = models.resnet50(pretrained=True)
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = model(image_tensor).squeeze(0)
    features=features.numpy()
    return features
def dataset_to_array():
    dataset_path = "../dataset"
    feature_vectors ={}
    classifications ={}
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            classifications[class_name] = []
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                features = extract_features(image_path)
                feature_vectors[image_name] = features
                classifications[class_name].append(image_name)
    return feature_vectors,classifications
def dataset_to_array2():
    dataset_path="../dataset"
    feature_vectors={}
    classifications={}
    classifications["1"]=[]
    for image_name in os.listdir(dataset_path):
        image_path="../dataset/"+image_name
        features = extract_features(image_path)
        feature_vectors[image_name] = features
        classifications["1"].append(image_name)
    return feature_vectors,classifications
def centroids(feature_vectors,classifications):
    class_centroids = {}
    for class_name, img_names in classifications.items():
        features = np.array([feature_vectors[name] for name in img_names])
        class_centroids[class_name] = features.mean(axis=0)
    return class_centroids
