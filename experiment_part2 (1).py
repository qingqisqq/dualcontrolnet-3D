import os
import cv2
import torch
import random
import numpy as np
import einops
import pandas as pd
from glob import glob
from tqdm import tqdm

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.hed import HEDdetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import config

# Define project directory and change to it. This is crucial for
# locating all relative paths correctly.
project_dir = './dualcontrolnet3d'
try:
    os.chdir(project_dir)
    print(f"Successfully changed working directory to: {os.getcwd()}")
except FileNotFoundError:
    print(f"Error: The directory '{project_dir}' was not found.")
    print("Please make sure the folder exists and run the script from the correct location.")
    exit()

# --- Part 1: Utility Functions ---

def construct_output_filename(prompt_row, xtile, ytile):
    """
    Constructs a descriptive output filename based on image metadata.
    
    Args:
        prompt_row (pd.Series): A row from the pandas DataFrame containing prompt data.
        xtile (str or int): The X-coordinate of the tile.
        ytile (str or int): The Y-coordinate of the tile.
        
    Returns:
        str: The formatted filename string.
    """
    idx = int(prompt_row['idx'])
    var = prompt_row['modified_variable']
    val = round(float(prompt_row['value']), 6)
    lat = round(float(prompt_row['latitude']), 4)
    lon = round(float(prompt_row['longitude']), 4)
    return f"generated_{idx}_{var}_{val}_{lat}_{lon}_{xtile}_{ytile}.png"

def save_image_to_disk(results, output_path):
    """
    Saves a generated image to the specified path.
    
    Args:
        results (list): A list containing the generated image data.
        output_path (str): The full path including the filename to save the image.
    """
    # The generated image is expected to be at the third position in the results list
    cv2.imwrite(output_path, results[2])
    
def set_seed(seed):
    """
    Sets the random seed.
    
    Args:
        seed (int): The seed number to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# --- Part 2: Main Processing Functions ---

def process_single_image(img_path, prompt, a_prompt, n_prompt, num_samples,
                         image_resolution, detect_resolution, ddim_steps,
                         guess_mode, strength, scale, seed, eta, model, ddim_sampler, config):
    """
    Processes a single input image using a given model to generate an output image
    based on a text prompt.
    
    Args:
        img_path (str): Path to the input image.
        prompt (str): The primary text prompt.
        a_prompt (str): The additional positive prompt.
        n_prompt (str): The negative prompt.
        num_samples (int): Number of samples to generate.
        image_resolution (int): Resolution for the output image.
        detect_resolution (int): Resolution for the detection map.
        ddim_steps (int): Number of DDIM diffusion steps.
        guess_mode (bool): Whether to use guess mode.
        strength (float): The control strength.
        scale (float): The unconditional guidance scale.
        seed (int): The random seed for generation.
        eta (float): The eta value for the DDIM sampler.
        model: The image generation model object.
        ddim_sampler: The DDIM sampler object.
        config: The model's configuration object.
        
    Returns:
        list: A list containing the detected map, input image, and generated images.
    """
    with torch.no_grad():
        # Load and resize the input image
        input_image = cv2.imread(img_path)
        input_image = cv2.resize(input_image, (image_resolution, image_resolution))
        H, W, C = input_image.shape

        # Load and resize the control map
        detected_map = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        detected_map = cv2.resize(detected_map, (detect_resolution, detect_resolution))

        # Handle images with an alpha channel (transparent areas)
        if detected_map.shape[2] == 4:
            trans_mask = detected_map[:, :, 3] == 0
            detected_map[trans_mask] = [255, 255, 255, 255]
            detected_map = cv2.cvtColor(detected_map, cv2.COLOR_BGRA2BGR)

        # Convert the control map to a torch tensor and prepare for batch processing
        control = cv2.cvtColor(detected_map, cv2.COLOR_BGR2RGB)
        control = torch.from_numpy(control).float().cuda() / 255.0
        control = torch.stack([control] * num_samples, dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        set_seed(seed)

        # Apply low VRAM shift if configured
        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        # Prepare conditional and unconditional inputs for the model
        cond = {
            "c_concat": [control],
            "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]
        }
        un_cond = {
            "c_concat": None if guess_mode else [control],
            "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]
        }

        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        # Set the control scales based on guess mode
        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else [strength] * 13

        # Run the DDIM sampler to generate the image
        samples, intermediates = ddim_sampler.sample(
            ddim_steps, num_samples, shape, cond, verbose=False, eta=eta,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond
        )

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        # Decode the generated latent space to a pixel image
        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        # Convert the generated images to BGR format for saving
        results = [cv2.cvtColor(x_samples[i], cv2.COLOR_RGB2BGR) for i in range(num_samples)]
        return [detected_map] + [input_image] + results

def process_all_test_images(test_folder, prompt_csv, output_dir, a_prompt, n_prompt, num_samples,
                            image_resolution, detect_resolution, ddim_steps, guess_mode,
                            strength, scale, eta, resume=True, model=None, ddim_sampler=None, config=None):
    """
    The main processing loop. It reads prompts from a CSV, iterates through
    input images, and generates new images based on the prompts.

    Args:
        test_folder (str): Path to the folder with input control images.
        prompt_csv (str): Path to the CSV file containing prompts and metadata.
        output_dir (str): Directory to save the generated images.
        ... (other parameters for image generation)
        resume (bool): If True, skips processing images that already exist in the output directory.
    
    Returns:
        list: A list of dictionaries with metadata for all generated images.
    """
    os.makedirs(output_dir, exist_ok=True)
    prompts_df = pd.read_csv(prompt_csv)
    
    # Filter for a specific set of indices
    target_idxs = [13954, 14902, 5506, 5854]#replace with idx
    prompts_df = prompts_df[prompts_df['idx'].isin(target_idxs)]

    # Get a list of all input image files for the target indices
    test_images = []
    for idx in target_idxs:
        test_images.extend(glob(os.path.join(test_folder, f"control_{idx}_*.png")))

    processed_count = 0
    skipped_count = 0
    all_results = []

    existing_files = set()
    if resume:
        # Check for existing generated files to resume the process
        existing_files = set(os.path.basename(f) for f in glob(os.path.join(output_dir, "generated_*.png")))
        print(f"Resuming process: Found {len(existing_files)} existing images to skip.")

    for img_path in tqdm(test_images, desc="Processing tile map generation"):
        filename = os.path.basename(img_path)
        parts = filename.replace("control_", "").replace(".png", "").split("_")
        if len(parts) < 3:
            continue
        idx, xtile, ytile = parts[0], parts[1], parts[2]

        # Find the corresponding prompt for the current image index
        for _, prompt_row in prompts_df.iterrows():
            if int(idx) != int(prompt_row['idx']):
                continue

            output_filename = construct_output_filename(prompt_row, xtile, ytile)
            output_path = os.path.join(output_dir, output_filename)

            if resume and output_filename in existing_files:
                skipped_count += 1
                continue

            prompt = prompt_row['prompt']
            seed = random.randint(0, 65535)

            try:
                results = process_single_image(img_path, prompt, a_prompt, n_prompt, num_samples,
                                               image_resolution, detect_resolution, ddim_steps,
                                               guess_mode, strength, scale, seed, eta, model, ddim_sampler, config)

                save_image_to_disk(results, output_path)

                all_results.append({
                    'output_path': output_path,
                    'idx': idx,
                    'tile': f"{xtile}_{ytile}",
                    'variable': prompt_row['modified_variable'],
                    'value': prompt_row['value'],
                    'lat': prompt_row['latitude'],
                    'lon': prompt_row['longitude'],
                    'prompt': prompt
                })

                processed_count += 1

            except Exception as e:
                print(f"Error: {e} @ tile {xtile}_{ytile} with prompt: {prompt[:40]}...")
                with open(os.path.join(output_dir, "error_log.txt"), "a") as f:
                    f.write(f"Error: {e} @ {output_filename}\n")

    print(f"Processing complete: Generated {processed_count} images, skipped {skipped_count} existing images.")
    return all_results

def count_processed_images(output_dir):
    """
    Counts the number of generated images in a specified directory.
    
    Args:
        output_dir (str): The directory to count files in.
        
    Returns:
        int: The number of files matching the "generated_*.png" pattern.
    """
    return len(glob(os.path.join(output_dir, "generated_*.png")))

# --- Part 3: Main Execution Block ---

def main():
    """
    The main function to orchestrate the entire process.
    """
    print("--- Starting image generation and processing script ---")

    # This part loads the actual model based on your configuration.
    version = '5265894'#this has to be in right folder
    epoch = '2'
    step = '22913'
    model = create_model('./models/cldm_v15.yaml').cpu()
    #Replace with trained model with your name
    model.load_state_dict(load_state_dict('./lightning_logs/version_'+version+'/checkpoints/epoch='+epoch+'-step='+step+'.ckpt', location='cuda'))

    model = model.cuda()
    ddim_sampler = DDIMSampler(model)
    
    # Define parameters for the process
    params = {
        'test_folder': './testcombinedcontrol_mapsgridc180',
        'prompt_csv': './metrics_datagridc180/m6sen.csv',
        'output_dir': "./m6sens",
        'a_prompt': "best quality, extremely detailed",
        'n_prompt': "lowres, cropped, worst quality, low quality, bad anatomy",
        'num_samples': 1,
        'image_resolution': 512,
        'detect_resolution': 512,
        'ddim_steps': 40,
        'guess_mode': False,
        'strength': 1.0,
        'scale': 9.0,
        'eta': 0.0,
        'resume': True
    }

    # Run the main image processing function
    all_results = process_all_test_images(**params, model=model, ddim_sampler=ddim_sampler, config=config)

    # Get the final count of generated images and print a summary
    final_count = count_processed_images(params['output_dir'])
    print(f"Total generated images: {final_count}, saved in: {os.path.abspath(params['output_dir'])}")

if __name__ == "__main__":
    main()
