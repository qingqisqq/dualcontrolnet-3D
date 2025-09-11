import numpy as np
import cv2
from scipy import ndimage
from collections import defaultdict
import os
import re
from tqdm import tqdm

def evaluate_land_use_image(image_path, color_similarity_threshold=30):
    """
    Evaluate a land use image and return metrics without visualization
    """
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return None
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Get image dimensions
    height, width, _ = img.shape
    total_pixels = height * width

    # Downsample for efficiency
    scale_factor = 0.2
    downsampled = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)
    pixels = downsampled.reshape(-1, 3)

    # Color quantization
    color_counts = defaultdict(int)
    quantization_factor = 40

    for pixel in pixels:
        quantized = tuple((pixel // quantization_factor) * quantization_factor)
        color_counts[quantized] += 1

    # Filter significant colors
    min_pixel_threshold = len(pixels) * 0.01
    significant_colors = []
    significant_counts = []

    for color, count in sorted(color_counts.items(), key=lambda x: -x[1]):
        if count >= min_pixel_threshold and np.mean(color) <= 240:
            significant_colors.append(np.array(color))
            significant_counts.append(count)

    # Calculate building coverage
    building_mask = np.zeros((height, width), dtype=np.uint8)
    
    pink_mask = np.all((img >= np.array([180, 100, 100])) & 
                      (img <= np.array([255, 180, 180])), axis=2)
    yellow_mask = np.all((img >= np.array([180, 150, 50])) & 
                        (img <= np.array([255, 230, 150])), axis=2)
    
    building_mask = np.logical_or(pink_mask, yellow_mask).astype(np.uint8)
    building_coverage_ratio = np.sum(building_mask) / total_pixels

    # Calculate block sizes
    avg_block_sizes = []
    min_block_size_threshold = 600

    for color in significant_colors:
        color_lower = np.maximum(0, color - color_similarity_threshold)
        color_upper = np.minimum(255, color + color_similarity_threshold)
        color_mask = np.all((img >= color_lower) & (img <= color_upper), axis=2).astype(np.uint8)
        
        labeled_mask, num_features = ndimage.label(color_mask)
        if num_features > 0:
            region_sizes = ndimage.sum(color_mask, labeled_mask, range(1, num_features + 1))
            valid_regions = region_sizes[region_sizes > min_block_size_threshold]
            if len(valid_regions) > 0:
                avg_block_sizes.append(np.mean(valid_regions))

    overall_avg_block_size = np.mean(avg_block_sizes) if avg_block_sizes else 0

    return {
        "diversity_score": len(significant_colors),
        "density_score": building_coverage_ratio,
        "design_score": overall_avg_block_size,
        "color_distribution": [count/sum(significant_counts) for count in significant_counts]
    }

def extract_idx(filename):
    """
    从文件名中提取 idx
    
    支持的文件名格式：
    - osm_map_10075_28.4763_-81.536797.png
    - osm_generated_12613_41.8615_-88.052194_0.png
    """
    patterns = [
        r'_(\d+)_\d+\.\d+_-?\d+\.\d+',  # 匹配下划线后的第一个数字
        r'(\d+)_\d+\.\d+_-?\d+\.\d+'   # 备选模式
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            try:
                return int(match.group(1))
            except:
                continue
    
    return None

def calculate_folder_similarities(folder1_path, folder2_path):
    """
    Calculate similarities between images in two folders with matching indices
    """
    # Get all images in both folders
    images1 = os.listdir(folder1_path)
    images2 = os.listdir(folder2_path)
    
    # Extract indices
    indices1 = {extract_idx(f): f for f in images1 if extract_idx(f) is not None}
    indices2 = {extract_idx(f): f for f in images2 if extract_idx(f) is not None}
    
    # Find common indices
    common_indices = set(indices1.keys()) & set(indices2.keys())
    
    print(f"Found {len(common_indices)} image pairs to process")
    
    results = {}
    overall_similarities = []

    for idx in tqdm(common_indices):
        filename1 = indices1[idx]
        filename2 = indices2[idx]
        
        img1_path = os.path.join(folder1_path, filename1)
        img2_path = os.path.join(folder2_path, filename2)
        
        # Evaluate both images
        metrics1 = evaluate_land_use_image(img1_path)
        metrics2 = evaluate_land_use_image(img2_path)
        
        if metrics1 is None or metrics2 is None:
            continue

        # Calculate similarities
        # Diversity similarity (normalized difference)
        diversity_sim = 1 - abs(metrics1["diversity_score"] - metrics2["diversity_score"]) / max(metrics1["diversity_score"], metrics2["diversity_score"])
        
        # Density similarity (absolute difference)
        density_sim = 1 - abs(metrics1["density_score"] - metrics2["density_score"])
        
        # Design similarity (normalized difference)
        max_block_size = max(metrics1["design_score"], metrics2["design_score"])
        design_sim = 1 - abs(metrics1["design_score"] - metrics2["design_score"]) / max_block_size if max_block_size > 0 else 1
        
        # Distribution similarity (minimum length of distributions)
        min_len = min(len(metrics1["color_distribution"]), len(metrics2["color_distribution"]))
        if min_len > 0:
            ref_dist = np.array(metrics1["color_distribution"][:min_len])
            comp_dist = np.array(metrics2["color_distribution"][:min_len])
            distribution_sim = 1 - np.mean(np.abs(ref_dist - comp_dist))
        else:
            distribution_sim = 0

        # Store results for this pair
        pair_result = {
            "image1": filename1,
            "image2": filename2,
            "diversity_similarity": diversity_sim,
            "density_similarity": density_sim,
            "design_similarity": design_sim,
            "distribution_similarity": distribution_sim,
            "overall_similarity": (diversity_sim + density_sim + design_sim + distribution_sim) / 4
        }
        
        results[f"idx_{idx}"] = pair_result
        overall_similarities.append(pair_result["overall_similarity"])

    # Calculate average similarities across all pairs
    average_results = {
        "average_diversity_similarity": np.mean([r["diversity_similarity"] for r in results.values()]),
        "average_density_similarity": np.mean([r["density_similarity"] for r in results.values()]),
        "average_design_similarity": np.mean([r["design_similarity"] for r in results.values()]),
        "average_distribution_similarity": np.mean([r["distribution_similarity"] for r in results.values()]),
        "overall_average_similarity": np.mean(overall_similarities)
    }

    return results, average_results

# Example usage
if __name__ == "__main__":
    folder1_path = "./512test_osm_mapsgrido150"
    folder2_path = "./m2o_test_results"
    
    individual_results, average_results = calculate_folder_similarities(folder1_path, folder2_path)
    
    # Print individual results
    print("\nIndividual Image Pair Similarities:")
    for idx, result in individual_results.items():
        print(f"\n{idx}:")
        print(f"Images: {result['image1']} <-> {result['image2']}")
        print(f"Diversity Similarity: {result['diversity_similarity']:.4f}")
        print(f"Density Similarity: {result['density_similarity']:.4f}")
        print(f"Design Similarity: {result['design_similarity']:.4f}")
        print(f"Distribution Similarity: {result['distribution_similarity']:.4f}")
        print(f"Overall Similarity: {result['overall_similarity']:.4f}")
    
    # Print average results
    print("\nAverage Similarities:")
    for metric, value in average_results.items():
        print(f"{metric}: {value:.4f}")
