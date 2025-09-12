import numpy as np
import os
import time
import torch
from torchvision import models, transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import random
import shutil
from scipy.linalg import sqrtm

# --- FID Calculation Utilities ---

class ImageDataset(Dataset):
    """
    A custom PyTorch Dataset for loading images from a directory.
    
    Attributes:
        path (str): The directory path containing the images.
        files (list): A list of image filenames found in the directory.
        transform (callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, path, transform=None):
        self.path = path
        # Filter for common image file extensions
        self.files = [f for f in os.listdir(path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform
        
    def __len__(self):
        """Returns the number of images in the dataset."""
        return len(self.files)
    
    def __getitem__(self, idx):
        """Loads and returns an image, applying the specified transform."""
        img_path = os.path.join(self.path, self.files[idx])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

def extract_features(dataloader, model, device):
    """
    Extracts features from a dataset using a pre-trained model.

    Args:
        dataloader (DataLoader): PyTorch DataLoader for the dataset.
        model (torch.nn.Module): The feature extraction model (e.g., InceptionV3).
        device (torch.device): The device (CPU or GPU) to run the computation on.

    Returns:
        np.ndarray: A concatenated array of features for all images.
    """
    # Set the model to evaluation mode
    model.eval()
    features = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move the batch to the specified device
            batch = batch.to(device)
            # Pass the batch through the model to get features
            feat = model(batch)
            features.append(feat.cpu().numpy())
    
    # Concatenate features from all batches into a single numpy array
    features = np.concatenate(features, axis=0)
    return features

def calculate_fid(real_features, gen_features):
    """
    Calculates the Frechet Inception Distance (FID) between two sets of features.

    The FID score measures the distance between the feature distributions of real
    and generated images. A lower FID score indicates higher quality and similarity.

    Args:
        real_features (np.ndarray): Features extracted from real images.
        gen_features (np.ndarray): Features extracted from generated images.

    Returns:
        float: The calculated FID score.
    """
    # Calculate the mean of the features
    mu1 = np.mean(real_features, axis=0)
    mu2 = np.mean(gen_features, axis=0)
    
    # Calculate the covariance matrices
    sigma1 = np.cov(real_features, rowvar=False)
    sigma2 = np.cov(gen_features, rowvar=False)
    
    # Calculate the squared difference between the means
    sum_sq_diff = np.sum((mu1 - mu2)**2)
    
    # Calculate the square root of the product of the covariance matrices
    try:
        covmean = sqrtm(sigma1 @ sigma2)
        # Check for imaginary components and take the real part
        if np.iscomplexobj(covmean):
            print("Warning: Covariance matrix square root has imaginary components. Using real part.")
            covmean = covmean.real
    except ValueError as e:
        print(f"Error calculating matrix square root: {e}")
        return np.inf

    # Calculate the FID score using the formula
    fid = sum_sq_diff + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    
    return fid

# --- Main Execution Flow ---
def main():
    """
    Main function to orchestrate the FID calculation process.
    It prepares the data, extracts features, and computes the FID score.
    """
    # Define the paths for the real and generated image folders
    real_merged_dir = "./merged6_real_images"
    gen_merged_dir = "./merged66_generated_images"
    
    # Create temporary directories to ensure balanced datasets
    temp_real_dir = "./temp_real_images"
    temp_gen_dir = "./temp_gen_images"
    
    try:
        print("Preparing datasets for FID calculation...")
        # Create temporary directories
        os.makedirs(temp_real_dir, exist_ok=True)
        os.makedirs(temp_gen_dir, exist_ok=True)
        
        # Get all image files from both directories
        real_images = [f for f in os.listdir(real_merged_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        gen_images = [f for f in os.listdir(gen_merged_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        # Determine the minimum number of images to ensure equal sample size
        min_count = min(len(real_images), len(gen_images))
        
        # Randomly select a balanced set of images from each folder
        selected_real = random.sample(real_images, min_count)
        selected_gen = random.sample(gen_images, min_count)
        
        # Copy the selected images to the temporary directories
        for img in selected_real:
            shutil.copy(os.path.join(real_merged_dir, img), os.path.join(temp_real_dir, img))
        
        for img in selected_gen:
            shutil.copy(os.path.join(gen_merged_dir, img), os.path.join(temp_gen_dir, img))
        
        print(f"Successfully balanced the number of images to {min_count} per folder.")
        
        # Set the device for computation (GPU if available, otherwise CPU)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load a pre-trained InceptionV3 model, a common choice for FID
        model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
        # Remove the final classification layer to use the model for feature extraction
        model.fc = torch.nn.Identity()
        model = model.to(device)
        
        # Define the image transformations required by the InceptionV3 model
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Create datasets and dataloaders for both image sets
        real_dataset = ImageDataset(temp_real_dir, transform)
        gen_dataset = ImageDataset(temp_gen_dir, transform)
        
        real_loader = DataLoader(real_dataset, batch_size=32, shuffle=False, num_workers=4)
        gen_loader = DataLoader(gen_dataset, batch_size=32, shuffle=False, num_workers=4)
        
        # Record the start time to measure performance
        start_time = time.time()
        
        # Extract features from both real and generated images
        print("Extracting features from real images...")
        real_features = extract_features(real_loader, model, device)
        print("Extracting features from generated images...")
        gen_features = extract_features(gen_loader, model, device)
        
        # Calculate the final FID score
        print("Calculating FID score...")
        fid_value = calculate_fid(real_features, gen_features)
        
        # Calculate and display the total computation time
        elapsed_time = time.time() - start_time
        
        print(f"\nFinal FID Score: {fid_value:.4f}")
        print(f"Total computation time: {elapsed_time:.2f} seconds")
    
    finally:
        # Clean up by removing the temporary directories
        print("\nCleaning up temporary directories...")
        if os.path.exists(temp_real_dir):
            shutil.rmtree(temp_real_dir)
        if os.path.exists(temp_gen_dir):
            shutil.rmtree(temp_gen_dir)
        print("Cleanup complete.")

if __name__ == "__main__":
    main()
