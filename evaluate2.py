import os
import cv2
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import wasserstein_distance
from tqdm import tqdm
import re

def calculate_gradient_distribution_similarities(img1_path, img2_path):
    # Load images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        return None
    
    # Convert to RGB
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    # Resize if dimensions don't match
    if img1_rgb.shape != img2_rgb.shape:
        img2_rgb = cv2.resize(img2_rgb, (img1_rgb.shape[1], img1_rgb.shape[0]))
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2GRAY)
    
    # Calculate Sobel gradients
    sobelx1 = cv2.Sobel(gray1, cv2.CV_64F, 1, 0, ksize=3)
    sobely1 = cv2.Sobel(gray1, cv2.CV_64F, 0, 1, ksize=3)
    sobelx2 = cv2.Sobel(gray2, cv2.CV_64F, 1, 0, ksize=3)
    sobely2 = cv2.Sobel(gray2, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate gradient magnitude
    grad1_magnitude = np.sqrt(sobelx1**2 + sobely1**2)
    grad2_magnitude = np.sqrt(sobelx2**2 + sobely2**2)
    
    # Calculate gradient direction
    grad1_direction = np.arctan2(sobely1, sobelx1)
    grad2_direction = np.arctan2(sobely2, sobelx2)
    
    # 1. Gradient Magnitude Distribution Similarity
    # Create histograms of gradient magnitudes
    hist1_mag = np.histogram(grad1_magnitude, bins=50, density=True)[0]
    hist2_mag = np.histogram(grad2_magnitude, bins=50, density=True)[0]
    
    # Normalize histograms
    hist1_mag = hist1_mag / np.sum(hist1_mag)
    hist2_mag = hist2_mag / np.sum(hist2_mag)
    
    # Calculate Earth Mover's Distance (Wasserstein distance)
    magnitude_dist_similarity = 1 / (1 + wasserstein_distance(hist1_mag, hist2_mag))
    
    # 2. Gradient Direction Distribution Similarity
    # Convert angles to degrees and shift to [0, 360]
    angles1 = (grad1_direction * 180 / np.pi) % 360
    angles2 = (grad2_direction * 180 / np.pi) % 360
    
    # Create orientation histograms (36 bins for 10-degree intervals)
    hist1_dir = np.histogram(angles1, bins=36, range=(0, 360), density=True)[0]
    hist2_dir = np.histogram(angles2, bins=36, range=(0, 360), density=True)[0]
    
    # Weight by gradient magnitude
    hist1_dir = hist1_dir * np.mean(grad1_magnitude)
    hist2_dir = hist2_dir * np.mean(grad2_magnitude)
    
    # Normalize histograms
    hist1_dir = hist1_dir / np.sum(hist1_dir)
    hist2_dir = hist2_dir / np.sum(hist2_dir)
    
    # Calculate cosine similarity for direction distribution
    direction_dist_similarity = 1 - cosine(hist1_dir, hist2_dir)
    
    # 3. Combined Gradient Distribution Similarity
    # Weight both similarities equally
    combined_similarity = (magnitude_dist_similarity + direction_dist_similarity) / 2
    
    return {
        'magnitude_distribution_similarity': float(magnitude_dist_similarity),
        'direction_distribution_similarity': float(direction_dist_similarity),
        'combined_distribution_similarity': float(combined_similarity)
    }

def process_image_folders(folder_path):
    # 获取所有图片
    all_images = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    
    # 分别获取遥感和OSM图片
    satellite_images = [f for f in all_images if f.startswith('satellite_generated_')]
    osm_images = [f for f in all_images if f.startswith('osm_generated_')]
    
    results = {}
    all_metrics = {}
    
    # 提取索引的函数
    def extract_idx(filename):
        # 使用正则表达式提取idx
        if filename.startswith('satellite_generated_'):
            match = re.search(r'satellite_generated_(\d+)_', filename)
        else:
            match = re.search(r'osm_generated_(\d+)_', filename)
        return int(match.group(1)) if match else None
    
    # 提取索引
    satellite_indices = {extract_idx(f): f for f in satellite_images if extract_idx(f) is not None}
    osm_indices = {extract_idx(f): f for f in osm_images if extract_idx(f) is not None}
    
    # 找到共同的索引
    common_indices = set(satellite_indices.keys()) & set(osm_indices.keys())
    
    print(f"Found {len(common_indices)} image pairs to process")
    
    # 处理每对匹配的图片
    for idx in tqdm(common_indices):
        satellite_filename = satellite_indices[idx]
        osm_filename = osm_indices[idx]
        
        satellite_path = os.path.join(folder_path, satellite_filename)
        osm_path = os.path.join(folder_path, osm_filename)
        
        similarities = calculate_gradient_distribution_similarities(satellite_path, osm_path)
        if similarities is not None:
            pair_key = f"{satellite_filename} -> {osm_filename}"
            results[pair_key] = similarities
            
            # 累积指标以计算平均值
            for metric, value in similarities.items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)
    
    # 计算平均值
    averages = {metric: np.mean(values) for metric, values in all_metrics.items()}
    
    return results, averages

# 使用示例
folder_path = "./mm3ofid_test_resultsextracted"  # 包含遥感和OSM图像的文件夹路径

# 确保输出文件夹存在
os.makedirs(folder_path, exist_ok=True)

print(f"Using folder: {folder_path}/")

# 计算相似度
similarities, averages = process_image_folders(folder_path)

# 打印结果
print("\nIndividual similarities:")
for pair, metrics in similarities.items():
    print(f"\nImage pair {pair}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

print("\nAverage metrics across all pairs:")
for metric, value in averages.items():
    print(f"{metric}: {value:.4f}")
