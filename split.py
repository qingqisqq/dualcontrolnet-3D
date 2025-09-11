import os
import cv2
import glob

# Input and output folder paths
input_folder = "./mm3ofid_test_results/"
output_folder = "./mm3ofid_test_resultsextracted/"  

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created output folder: {output_folder}")

# Get all images to process
input_files = glob.glob(os.path.join(input_folder, "generated_*.png"))
print(f"Found {len(input_files)} images to process")

processed_count = 0
for img_path in input_files:
    try:
        # Get original filename without extension
        filename = os.path.basename(img_path)
        filename_without_ext = os.path.splitext(filename)[0]  # removes .png
        
        # Read image
        img = cv2.imread(img_path)
        height, width = img.shape[:2]
        
        # Split image into left and right halves from the middle
        mid_point = width // 2
        left_half = img[:, 0:mid_point]
        right_half = img[:, mid_point:width]
        
        # Calculate new dimensions - only double the width, keep same height
        new_width = mid_point * 2
        
        # Stretch both halves to twice their original width
        # Using INTER_CUBIC for better quality upsampling
        left_half_stretched = cv2.resize(left_half, (new_width, height), interpolation=cv2.INTER_CUBIC)
        right_half_stretched = cv2.resize(right_half, (new_width, height), interpolation=cv2.INTER_CUBIC)
        
        # Create new filenames by adding prefixes to original filename
        satellite_filename = f"satellite_{filename}"
        osm_filename = f"osm_{filename}"
        
        # Save processed images to the same output folder
        cv2.imwrite(os.path.join(output_folder, satellite_filename), left_half_stretched)
        cv2.imwrite(os.path.join(output_folder, osm_filename), right_half_stretched)
        
        processed_count += 1
        print(f"Processed: {filename} -> {satellite_filename} and {osm_filename}")
    except Exception as e:
        print(f"Error processing image {filename}: {e}")

print(f"Processing complete! Processed {processed_count} images")
