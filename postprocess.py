import numpy as np
from PIL import Image
import os
from glob import glob
from tqdm import tqdm

# Define standard color dictionary for land use categories
COLOR_MAPPING = {
    'residential': (255, 202, 101),
    'commercial': (255, 128, 128),
    'industrial': (191, 191, 191),
    'retail': (255, 85, 85),
    'parking': (239, 239, 239),
    'school': (255, 240, 170),
    'university': (255, 240, 170),
    'hospital': (255, 230, 230),
    'park': (178, 216, 178),
    'garden': (178, 216, 178),
    'recreation_ground': (184, 230, 184),
    'playground': (204, 230, 204),
    'sports_centre': (204, 230, 204),
    'stadium': (204, 230, 204),
    'pitch': (204, 230, 204),
    'golf_course': (178, 216, 178),
    'forest': (140, 191, 140),
    'wood': (140, 191, 140),
    'grass': (204, 255, 204),
    'grassland': (204, 255, 204),
    'meadow': (204, 255, 204),
    'heath': (204, 230, 204),
    'scrub': (184, 230, 184),
    'wetland': (186, 230, 230),
    'water': (179, 217, 255),
    'beach': (255, 245, 204),
    'sand': (255, 245, 204),
    'farmland': (255, 255, 204),
    'orchard': (230, 255, 179),
    'vineyard': (230, 255, 179),
    'cemetery': (209, 207, 205),
    'white': (255, 255, 255)  
}

def classify_pixel(pixel, color_mapping):
    """
    Classify a pixel to the closest standard color using Euclidean distance
    
    Args:
        pixel: Input pixel RGB values
        color_mapping: Dictionary of standard colors
    
    Returns:
        RGB values of the closest standard color
    """
    min_distance = float('inf')
    closest_category = None
    
    # Calculate distance to each standard color
    for category, standard_color in color_mapping.items():
        # Calculate Euclidean distance between pixel and standard color
        distance = np.sqrt(sum((p - s) ** 2 for p, s in zip(pixel, standard_color)))
        
        # Update if this is the closest color so far
        if distance < min_distance:
            min_distance = distance
            closest_category = category
    
    return COLOR_MAPPING[closest_category]

def convert_to_standard_colors(input_path, output_path):
    """
    Convert input land use image to standardized colors
    
    Args:
        input_path: Path to input image file
        output_path: Path where output image will be saved
    """
    try:
        # Read input image
        img = Image.open(input_path)
        img_array = np.array(img)
        
        # Create output array with same dimensions
        output_array = np.zeros_like(img_array)
        
        # Process each pixel in the image
        height, width = img_array.shape[:2]
        for y in range(height):
            for x in range(width):
                pixel = img_array[y, x]
                # Convert pixel to nearest standard color
                standard_color = classify_pixel(pixel, COLOR_MAPPING)
                output_array[y, x] = standard_color
        
        # Save the standardized image
        output_img = Image.fromarray(output_array.astype('uint8'))
        output_img.save(output_path)
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return False

def process_folder(input_folder, output_folder):
    """
    Process all osm_ images in the input folder
    
    Args:
        input_folder: Path to folder containing input images
        output_folder: Path to folder where output images will be saved
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all osm_ images in input folder
    osm_images = glob(os.path.join(input_folder, "osm_*.png"))
    
    # Process each image
    successful = 0
    failed = 0
    
    print(f"Found {len(osm_images)} images to process")
    
    for input_path in tqdm(osm_images, desc="Processing images"):
        # Create output path
        filename = os.path.basename(input_path)
        output_path = os.path.join(output_folder, f"standardized_{filename}")
        
        # Process the image
        if convert_to_standard_colors(input_path, output_path):
            successful += 1
        else:
            failed += 1
    
    print(f"\nProcessing complete:")
    print(f"Successfully processed: {successful} images")
    print(f"Failed to process: {failed} images")

# Example usage
input_folder = "./m6fid_test_resultsextracted"  
output_folder = "./standardizedm6fid"  
process_folder(input_folder, output_folder)
