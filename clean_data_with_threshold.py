import cv2
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil

# ---
## Part 1: Image Analysis and Data Export

def calculate_white_ratio(image_path):
    """
    Calculates the ratio of white pixels (value 255) to the total pixels
    in a grayscale image.

    Args:
        image_path (str): The path to the image file.

    Returns:
        float or None: The white pixel ratio (a value between 0.0 and 1.0)
                       or None if the image cannot be read.
    """
    # Read the image in grayscale mode
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Warning: Could not read image at {image_path}. Skipping.")
        return None

    # Count pixels with a value of 255 (pure white)
    white_pixels = np.sum(img == 255)
    # Get the total number of pixels in the image
    total_pixels = img.size
    
    # Calculate the ratio
    white_ratio = white_pixels / total_pixels
    
    return white_ratio

def analyze_and_export(folder_path, threshold=0.5, output_csv='white_ratio_results.csv'):
    """
    Analyzes all PNG images in a specified folder to calculate their white
    pixel ratio. It saves the results to a CSV file and prints a summary.

    Args:
        folder_path (str): The directory containing the images.
        threshold (float): The threshold for the white ratio. Images with a ratio
                           above this value will be flagged.
        output_csv (str): The name of the output CSV file.
    """
    results = []
    
    # Get a list of all PNG files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    
    print(f"Starting analysis of images in {folder_path}...")
    
    # Use tqdm to show a progress bar
    for image_file in tqdm(image_files, desc="Processing Images"):
        # The filename format is expected to be 'osm_map_idx_lat_lon.png'
        # Extract metadata (idx, lat, lon) from the filename
        try:
            parts = image_file.replace("osm_map_", "").replace(".png", "").split("_")
            idx, lat, lon = parts
        except ValueError:
            print(f"Warning: Filename format not recognized for {image_file}. Skipping.")
            continue
        
        image_path = os.path.join(folder_path, image_file)
        white_ratio = calculate_white_ratio(image_path)
        
        if white_ratio is not None:
            # Append the data to the results list
            results.append({
                'idx': int(idx),
                'lat': float(lat),
                'lon': float(lon),
                'white_ratio': white_ratio,
                'exceed_threshold': white_ratio > threshold
            })
    
    # Create a pandas DataFrame from the results
    df = pd.DataFrame(results)
    
    # Save the DataFrame to a CSV file
    df.to_csv(output_csv, index=False)
    
    # Print a summary of the analysis
    filtered_count = df['exceed_threshold'].sum()
    print("\n--- Analysis Summary ---")
    print(f"Total images processed: {len(df)}")
    print(f"Images exceeding the {threshold} threshold: {filtered_count}")
    print(f"Results saved to {output_csv}")

# ---
## Part 2: File Movement Based on Analysis
def move_images(source_folder, target_folder, move_idx, folder_type):
    """
    Moves images from a source folder to a target folder if their
    index is in the list of indices to be moved.

    Args:
        source_folder (str): The path to the folder containing the source images.
        target_folder (str): The path to the destination folder.
        move_idx (list): A list of integer indices for the images to be moved.
        folder_type (str): A string indicating the type of folder (e.g., 'osm_maps').
    
    Returns:
        int: The number of files successfully moved.
    """
    # Create the target directory if it doesn't exist
    os.makedirs(target_folder, exist_ok=True)
    
    moved_count = 0
    
    # Get a list of all files in the source folder
    files = os.listdir(source_folder)
    
    print(f"\nProcessing folder: {source_folder}")

    for file in tqdm(files, desc=f"Moving from {folder_type}"):
        try:
            # Parse the filename to get the index.
            # Filename format varies by folder_type.
            if folder_type == 'osm_maps':
                # Filename format: osm_map_idx_lat_lon.png
                parts = file.replace("osm_map_", "").replace(".png", "").split('_')
                file_idx = int(parts[0])
            else:
                # Filename format: prefix_idx_lat_lon.png
                parts = file.split('_')
                file_idx = int(parts[1])
            
            # If the file's index is in the list of indices to move, move the file
            if file_idx in move_idx:
                source_path = os.path.join(source_folder, file)
                target_path = os.path.join(target_folder, file)
                shutil.move(source_path, target_path)
                moved_count += 1
                
        except Exception as e:
            # If there's an error parsing the filename, skip the file
            print(f"Skipped file: {file} (Error: {str(e)})")
            
    print(f"Moved {moved_count} files from {source_folder} to {target_folder}")
    return moved_count

# ---
## Part 3: Main Execution
def main():
    """
    The main function to run the entire script. It first analyzes images
    and then moves them based on the analysis results.
    """
    print("--- Starting image processing and file organization script ---")
    
    # Define project directory and change to it
    project_dir = './dualcontrolnet3d'
    if not os.path.isdir(project_dir):
        print(f"Error: Project directory '{project_dir}' not found.")
        return
        
    os.chdir(project_dir)
    print(f"Changed current working directory to: {os.getcwd()}")
    
    # Step 1: Analyze images and save the white ratio results
    osm_maps_folder_path = './osm_mapsgridc180'
    white_ratio_csv = 'white_ratio_results.csv'
    
    # Adjust this threshold as needed
    analysis_threshold = 0.7 
    
    analyze_and_export(osm_maps_folder_path, threshold=analysis_threshold, output_csv=white_ratio_csv)
    
    # Step 2: Move files based on the analysis results
    
    # Read the CSV file containing the white ratio information
    try:
        df = pd.read_csv(white_ratio_csv)
        # Sort by white_ratio in descending order and save
        df_sorted = df.sort_values(by='white_ratio', ascending=False)
        df_sorted.to_csv('white_ratio_results_sorted.csv', index=False)
        print("Results sorted by white ratio and saved to 'white_ratio_results_sorted.csv'.")
    except FileNotFoundError:
        print(f"Error: Could not find the analysis results file '{white_ratio_csv}'.")
        return
        
    # Get the indices of images with a white ratio greater than 0.7
    move_ratio_threshold = 0.7
    move_idx = df[df['white_ratio'] > move_ratio_threshold]['idx'].tolist()
    
    if not move_idx:
        print(f"No images found with a white ratio above {move_ratio_threshold}. No files will be moved.")
        return
    
    # Define the source and target folders for all image types
    source_folders = {
        'satellite': './satellite_imagesgridc180',
        'osm_maps': './osm_mapsgridc180',
        'control': './control_mapsgridc180'
    }

    target_folders = {
        'satellite': f'./satellite_imagesgridc180_above_{move_ratio_threshold}',
        'osm_maps': f'./osm_mapsgridc180_above_{move_ratio_threshold}',
        'control': f'./control_mapsgridc180_above_{move_ratio_threshold}'
    }

    # Process and move images for each folder type
    total_moved = 0
    for folder_type, source_folder in source_folders.items():
        target_folder = target_folders[folder_type]
        moved_count = move_images(source_folder, target_folder, move_idx, folder_type)
        total_moved += moved_count

    # Print final summary statistics
    print("\n--- Process Complete ---")
    print(f"Number of unique image IDs to move: {len(move_idx)}")
    print(f"Total files moved across all folders: {total_moved}")
    print("Verification of file counts:")
    for folder_type in source_folders:
        source_count = len(os.listdir(source_folders[folder_type]))
        target_count = len(os.listdir(target_folders[folder_type]))
        print(f"  {folder_type} original folder remaining files: {source_count}")
        print(f"  {folder_type} moved files folder count: {target_count}")

if __name__ == "__main__":
    main()
