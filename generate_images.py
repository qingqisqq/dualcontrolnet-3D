# =================================================================
# 1. IMPORTS
# All necessary libraries are imported at the beginning.
# =================================================================
import os
import cv2
import einops
import numpy as np
import torch
import random
import pandas as pd
from tqdm import tqdm
from glob import glob

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import config

# Imports from the local 'share.py' file
from share import *


# =================================================================
# 2. FUNCTION DEFINITIONS
# =================================================================

def process_all_test_images(model, ddim_sampler, test_folder, prompt_csv, output_dir, a_prompt, n_prompt, num_samples, 
                           image_resolution, detect_resolution, ddim_steps, guess_mode, 
                           strength, scale, eta):
    """
    Iterates through all images in a test folder, generates new images based on corresponding prompts from a CSV file,
    and saves the results.

    Args:
        model: The loaded ControlNet model.
        ddim_sampler: The DDIM sampler instance.
        test_folder (str): Path to the folder containing input control map images.
        prompt_csv (str): Path to the CSV file containing prompts.
        output_dir (str): Path to the directory where generated images will be saved.
        ... and other generation parameters.
    """
    
    # Ensure the output directory exists, create it if it doesn't.
    os.makedirs(output_dir, exist_ok=True)
    
    # Load prompts from the specified CSV file.
    try:
        prompts_df = pd.read_csv(prompt_csv)
        print(f"Loaded {len(prompts_df)} prompts from CSV.")
    except Exception as e:
        print(f"Error reading prompt CSV file: {e}")
        return []
    
    # Find all PNG images in the test folder.
    test_images = glob(os.path.join(test_folder, "*.png"))
    print(f"Found {len(test_images)} test images.")
    
    all_results = []
    skipped_images = []
    
    # Loop through each found image with a progress bar (tqdm).
    for img_path in tqdm(test_images, desc="Processing test images"):
        try:
            # Parse the filename to extract identifiers (idx, xtile, ytile).
            filename = os.path.basename(img_path)
            parts = filename.replace("control_map_", "").replace(".png", "").split("_")
            
            if len(parts) >= 3:
                idx, xtile, ytile = parts[0], parts[1], parts[2]
                
                # Find the corresponding prompt in the DataFrame using the 'idx'.
                prompt_row = prompts_df[prompts_df['idx'] == int(idx)]
                if not prompt_row.empty and 'prompt' in prompt_row.columns:
                    prompt = prompt_row['prompt'].values[0]
                    
                    # Generate a random seed for each image for varied results.
                    seed = random.randint(0, 65535)
                    
                    # Call the core function to process a single image.
                    results = process_single_image(model, ddim_sampler, img_path, prompt, a_prompt, n_prompt, num_samples,
                                                image_resolution, detect_resolution, ddim_steps,
                                                guess_mode, strength, scale, seed, eta, idx, xtile, ytile)
                    
                    if results is not None:
                        # Save the generated images to disk.
                        save_results(results, output_dir, idx, xtile, ytile)
                        
                        # Store results in memory (optional).
                        all_results.append({
                            'idx': idx,
                            'xtile': xtile,
                            'ytile': ytile,
                            'input_image': results[1],
                            'generated_image': results[2] # Assumes num_samples=1
                        })
                else:
                    skipped_images.append((filename, "No matching prompt found"))
            else:
                skipped_images.append((filename, "Invalid filename format"))
                
        except Exception as e:
            # Catch and report any errors during processing.
            print(f"Error processing image {img_path}: {e}")
            skipped_images.append((filename, str(e)))
            continue
    
    # Print a summary of the processing.
    print(f"\nProcessing complete:")
    print(f"Successfully processed: {len(all_results)} images")
    print(f"Skipped: {len(skipped_images)} images")
    
    # If any images were skipped, save a log file for debugging.
    if skipped_images:
        skip_log_path = os.path.join(output_dir, 'skipped_images.txt')
        with open(skip_log_path, 'w') as f:
            f.write("Skipped images and reasons:\n")
            for img, reason in skipped_images:
                f.write(f"{img}: {reason}\n")
        print(f"Skipped images log saved to: {skip_log_path}")
    
    return all_results


def process_single_image(model, ddim_sampler, img_path, prompt, a_prompt, n_prompt, num_samples, 
                        image_resolution, detect_resolution, ddim_steps, 
                        guess_mode, strength, scale, seed, eta, idx, xtile, ytile):
    """
    Generates an image from a single control map and text prompt.
    This is the core inference function.
    """
    
    # Use torch.no_grad() to disable gradient calculations, saving memory and speeding up inference.
    with torch.no_grad():
        # Read and resize the input image (used for visualization/comparison).
        input_image = cv2.imread(img_path)
        input_image = cv2.resize(input_image, (image_resolution, image_resolution))
        H, W, C = input_image.shape

        # Read the control map, keeping all channels (including alpha if present).
        detected_map = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        detected_map = cv2.resize(detected_map, (detect_resolution, detect_resolution))
        
        # Handle different image formats for the control map.
        if detected_map.ndim == 2:  # Grayscale image
            detected_map = cv2.cvtColor(detected_map, cv2.COLOR_GRAY2BGR)
        
        if detected_map.shape[2] == 4:  # Image with alpha channel (BGRA)
            # Make transparent areas white, then convert to 3-channel BGR.
            trans_mask = detected_map[:,:,3] == 0
            detected_map[trans_mask] = [255, 255, 255, 255]
            detected_map = cv2.cvtColor(detected_map, cv2.COLOR_BGRA2BGR)
        
        # Prepare the control tensor for the model.
        control = cv2.cvtColor(detected_map, cv2.COLOR_BGR2RGB) # OpenCV uses BGR, PyTorch expects RGB
        control = torch.from_numpy(control.copy()).float().cuda() / 255.0 # Convert to tensor, normalize, move to GPU
        control = torch.stack([control for _ in range(num_samples)], dim=0) # Create a batch
        control = einops.rearrange(control, 'b h w c -> b c h w').clone() # Rearrange dimensions to (B, C, H, W)

        # Set the seed for reproducibility.
        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        # Use low VRAM mode if enabled in config.
        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        # Prepare conditioning vectors (prompts and control map).
        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8) # Shape of the latent space tensor

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        # Set control scales (how much to adhere to the control map).
        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
        
        # Run the DDIM sampler to generate image latents.
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                   shape, cond, verbose=False, eta=eta,
                                                   unconditional_guidance_scale=scale,
                                                   unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        # Decode the generated latents back into pixel space.
        x_samples = model.decode_first_stage(samples)
        
        # Post-process the output tensor to a viewable image format (numpy array).
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        # Convert images from RGB back to BGR for saving with OpenCV.
        results = [cv2.cvtColor(x_samples[i], cv2.COLOR_RGB2BGR) for i in range(num_samples)]

        # Return a list containing the control map, original image, and generated results.
        return [detected_map] + [input_image] + results


def save_results(results, output_dir, idx, xtile, ytile):
    """Saves the generated images to the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    # The first two elements in 'results' are the detected_map and input_image.
    # We only save the generated images, which start from index 2.
    for i, img in enumerate(results[2:]):
        filename = f"generated_{idx}_{xtile}_{ytile}_{i}.png"
        cv2.imwrite(os.path.join(output_dir, filename), img)
    

def main():
    """The main entry point of the script."""
    
    # =================================================================
    # 3. MODEL INITIALIZATION
    # This section loads the pre-trained ControlNet model.
    # =================================================================
    print("Initializing model...")
    version = '5265894'  # IMPORTANT: replace with your model version
    epoch = '2'
    step = '22913'
    
    # Create model architecture from the YAML config file.
    model = create_model('./models/cldm_v15.yaml').cpu()
    # Load the trained weights from the checkpoint file.
    model.load_state_dict(load_state_dict(f'./lightning_logs/version_{version}/checkpoints/epoch={epoch}-step={step}.ckpt', location='cuda'))
    # Move the model to the GPU.
    model = model.cuda()
    # Initialize the DDIM sampler.
    ddim_sampler = DDIMSampler(model)
    print("Model loaded successfully.")
    
    # =================================================================
    # 4. PARAMETERS
    # Define all hyperparameters and paths for the generation process.
    # =================================================================
    params = {
        'test_folder': './controlmaphere',  # Input folder with control maps
        'prompt_csv': './metrics_datagridc180/testprompthere.csv',  # CSV with prompts
        'output_dir': "./mmx_test_results",  # Output folder for results
        'a_prompt': "best quality, extremely detailed",  # Appended to every positive prompt
        'n_prompt': "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",  # Negative prompt
        'num_samples': 1,           # Number of images to generate per prompt
        'image_resolution': 512,    # Resolution of the output image
        'detect_resolution': 512,   # Resolution to process the control map at
        'ddim_steps': 40,           # Number of diffusion steps
        'guess_mode': False,        # If True, the model "guesses" content without prompts
        'strength': 1.0,            # ControlNet strength
        'scale': 9.0,               # Classifier-Free Guidance scale
        'eta': 0.0,                 # DDIM parameter (0.0 for DDIM, 1.0 for DDPM)
    }
    
    # =================================================================
    # 5. EXECUTION
    # Start the main processing loop.
    # =================================================================
    process_all_test_images(model=model, ddim_sampler=ddim_sampler, **params)
    
    print(f"All processing complete. Results are saved in {params['output_dir']}")
    
#execute   
if __name__ == "__main__":
    main()
