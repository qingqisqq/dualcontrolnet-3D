import os
import cv2
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import wasserstein_distance
from tqdm import tqdm
import re

def calculate_gradient_distribution_similarities(img1_path, img2_path):
    """
    Calculates the similarity between the gradient distributions of two images.

    This function measures both the magnitude and direction distributions of gradients
    to quantify structural similarity.

    Args:
        img1_path (str): The file path to the first image.
        img2_path (str): The file path to the second image.

    Returns:
        dict: A dictionary containing 'magnitude_distribution_similarity',
              'direction_distribution_similarity', and a 'combined_distribution_similarity'
              score. Returns None if images cannot be loaded.
    """
    # Load images from the specified paths
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    # Check if images were loaded successfully
    if img1 is None or img2 is None:
        print(f"Warning: Could not load one or both images: {img1_path}, {img2_path}")
        return None
    
    # Convert images to RGB color space for consistent processing
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    # Resize the second image to match the dimensions of the first, if they differ
    if img1_rgb.shape != img2_rgb.shape:
        img2_rgb = cv2.resize(img2_rgb, (img1_rgb.shape[1], img1_rgb.shape[0]))
    
    # Convert RGB images to grayscale for gradient calculation
    gray1 = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2GRAY)
    
    # Calculate the Sobel x and y gradients for both grayscale images
    sobelx1 = cv2.Sobel(gray1, cv2.CV_64F, 1, 0, ksize=3)
    sobely1 = cv2.Sobel(gray1, cv2.CV_64F, 0, 1, ksize=3)
    sobelx2 = cv2.Sobel(gray2, cv2.CV_64F, 1, 0, ksize=3)
    sobely2 = cv2.Sobel(gray2, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate the gradient magnitude using the L2 norm (sqrt(x^2 + y^2))
    grad1_magnitude = np.sqrt(sobelx1**2 + sobely1**2)
    grad2_magnitude = np.sqrt(sobelx2**2 + sobely2**2)
    
    # Calculate the gradient direction (angle) using the arctan2 function
    grad1_direction = np.arctan2(sobely1, sobelx1)
    grad2_direction = np.arctan2(sobely2, sobelx2)
    
    # --- 1. Gradient Magnitude Distribution Similarity ---
    
    # Create histograms of gradient magnitudes
    hist1_mag = np.histogram(grad1_magnitude, bins=50, density=True)[0]
    hist2_mag = np.histogram(grad2_magnitude, bins=50, density=True)[0]
    
    # Normalize histograms to sum to 1
    hist1_mag = hist1_mag / np.sum(hist1_mag)
    hist2_mag = hist2_mag / np.sum(hist2_mag)
    
    # Calculate Earth Mover's Distance (Wasserstein distance)
    # A smaller distance implies greater similarity. We transform this into a similarity score (0 to 1).
    magnitude_dist_similarity = 1 / (1 + wasserstein_distance(hist1_mag, hist2_mag))
    
    # --- 2. Gradient Direction Distribution Similarity ---
    
    # Convert angles from radians to degrees and wrap them to the [0, 360] range
    angles1 = (grad1_direction * 180 / np.pi) % 360
    angles2 = (grad2_direction * 180 / np.pi) % 360
    
    # Create orientation histograms with 36 bins (for 10-degree intervals)
    hist1_dir = np.histogram(angles1, bins=36, range=(0, 360), density=True)[0]
    hist2_dir = np.histogram(angles2, bins=36, range=(0, 360), density=True)[0]
    
    # Weight the orientation histograms by the average gradient magnitude
    hist1_dir = hist1_dir * np.mean(grad1_magnitude)
    hist2_dir = hist2_dir * np.mean(grad2_magnitude)
    
    # Normalize the weighted histograms to sum to 1
    hist1_dir = hist1_dir / np.sum(hist1_dir)
    hist2_dir = hist2_dir / np.sum(hist2_dir)
    
    # Calculate the cosine similarity for the direction distributions
    # Cosine similarity is 1 for identical vectors and approaches 0 for dissimilar vectors.
    direction_dist_similarity = 1 - cosine(hist1_dir, hist2_dir)
    
    # --- 3. Combined Gradient Distribution Similarity ---
    
    # Combine the two similarity scores with equal weighting
    combined_similarity = (magnitude_dist_similarity + direction_dist_similarity) / 2
    
    return {
        'magnitude_distribution_similarity': float(magnitude_dist_similarity),
        'direction_distribution_similarity': float(direction_dist_similarity),
        'combined_distribution_similarity': float(combined_similarity)
    }

def process_image_folders(folder_path):
    """
    Processes a folder containing both satellite and OSM images, calculates the
    gradient distribution similarity for each pair, and returns the results.

    The function assumes images are named with a pattern like `*_idx_*.png`.

    Args:
        folder_path (str): The directory containing the image files.

    Returns:
        tuple: A tuple containing two dictionaries:
               - results (dict): Individual similarity scores for each image pair.
               - averages (dict): The average similarity scores across all pairs.
    """
    # Get a list of all .png files in the specified folder
    all_images = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    
    # Separate the images into satellite and OSM categories based on their filename prefix
    satellite_images = [f for f in all_images if f.startswith('satellite_generated_')]
    osm_images = [f for f in all_images if f.startswith('osm_generated_')]
    
    # Dictionaries to store results and metric values for averaging
    results = {}
    all_metrics = {}
    
    # Helper function to extract the numerical index (idx) from a filename using regex
    def extract_idx(filename):
        if filename.startswith('satellite_generated_'):
            match = re.search(r'satellite_generated_(\d+)_', filename)
        else:
            match = re.search(r'osm_generated_(\d+)_', filename)
        return int(match.group(1)) if match else None
    
    # Map filenames to their extracted indices
    satellite_indices = {extract_idx(f): f for f in satellite_images if extract_idx(f) is not None}
    osm_indices = {extract_idx(f): f for f in osm_images if extract_idx(f) is not None}
    
    # Find the common indices present in both satellite and OSM lists to form pairs
    common_indices = set(satellite_indices.keys()) & set(osm_indices.keys())
    
    print(f"Found {len(common_indices)} image pairs to process.")
    
    # Iterate over the common indices to process each corresponding image pair
    for idx in tqdm(common_indices):
        satellite_filename = satellite_indices[idx]
        osm_filename = osm_indices[idx]
        
        satellite_path = os.path.join(folder_path, satellite_filename)
        osm_path = os.path.join(folder_path, osm_filename)
        
        # Calculate the similarity metrics for the current image pair
        similarities = calculate_gradient_distribution_similarities(satellite_path, osm_path)
        
        if similarities is not None:
            pair_key = f"{satellite_filename} -> {osm_filename}"
            results[pair_key] = similarities
            
            # Accumulate the metric values to calculate the average later
            for metric, value in similarities.items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)
    
    # Calculate the average of each metric across all processed pairs
    averages = {metric: np.mean(values) for metric, values in all_metrics.items()}
    
    return results, averages

# --- Script Execution Example ---
if __name__ == "__main__":
    # Define the path to the folder containing the images
    folder_path = "./mm3ofid_test_resultsextracted"#get from generate_images.py
    
    # Ensure the specified output directory exists
    os.makedirs(folder_path, exist_ok=True)
    
    print(f"Using folder: {folder_path}/")
    
    # Run the main function to calculate the similarities
    similarities, averages = process_image_folders(folder_path)
    
    # Print the individual similarity results
    print("\nIndividual similarities:")
    for pair, metrics in similarities.items():
        print(f"\nImage pair {pair}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    
    # Print the average metric results for a summary
    print("\nAverage metrics across all pairs:")
    for metric, value in averages.items():
        print(f"{metric}: {value:.4f}")
