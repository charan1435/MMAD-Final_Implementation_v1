# mri_validator.py
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pickle

class MRIValidator:
    def __init__(self, model_name='resnet18', device=None):
        """Initialize the MRI validator with a pre-trained model"""
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"MRI Validator using device: {self.device}")
        
        # Load pre-trained model
        if model_name == 'resnet18':
            self.model = models.resnet18(weights='DEFAULT')
        elif model_name == 'resnet50':
            self.model = models.resnet50(weights='DEFAULT')
        elif model_name == 'efficientnet':
            self.model = models.efficientnet_b0(weights='DEFAULT')
        else:
            raise ValueError(f"Unsupported model: {model_name}")
            
        # Remove the classification layer
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize centroid and threshold
        self.mri_centroid = None
        self.distance_threshold = None
        
    def extract_features(self, image):
        """Extract features from an image"""
        if isinstance(image, str):
            # Load image from path
            try:
                image = Image.open(image).convert('RGB')
            except Exception as e:
                print(f"Error loading image {image}: {e}")
                return None
        
        if isinstance(image, Image.Image):
            # Apply transformations
            try:
                image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            except Exception as e:
                print(f"Error transforming image: {e}")
                return None
        else:
            # Assume it's already a properly formatted tensor
            image_tensor = image.to(self.device)
            
        # Extract features
        with torch.no_grad():
            try:
                features = self.feature_extractor(image_tensor)
                features = features.flatten().cpu().numpy()
                return features
            except Exception as e:
                print(f"Error extracting features: {e}")
                return None
    
    def is_brain_mri(self, image, threshold_multiplier=1.0):
        """
        Determine if an image is likely a brain MRI
        Returns (is_mri, confidence, distance)
        """
        if self.mri_centroid is None:
            raise ValueError("MRI centroid not computed. Load the validator first.")
            
        # Extract features
        features = self.extract_features(image)
        if features is None:
            return False, 0.0, float('inf')
        
        # Calculate distance to MRI centroid
        distance = np.linalg.norm(features - self.mri_centroid)
        
        # Apply threshold
        adjusted_threshold = self.distance_threshold * threshold_multiplier
        is_mri = distance < adjusted_threshold
        
        # Calculate confidence score (inverse of normalized distance)
        # Lower distance = higher confidence
        confidence = max(0, 1 - (distance / (2 * adjusted_threshold)))
        
        return is_mri, confidence, distance
    
    def load(self, filepath):
        """Load the MRI validator state from disk"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
            
        self.mri_centroid = state['mri_centroid']
        self.distance_threshold = state['distance_threshold']
        
        print(f"MRI validator loaded from {filepath}")
        print(f"Distance threshold: {self.distance_threshold:.4f}")
        
        if 'timestamp' in state:
            import datetime
            timestamp = datetime.datetime.fromtimestamp(state['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            print(f"Model created on: {timestamp}")
        
        return self