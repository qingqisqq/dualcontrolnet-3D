# dualcontrolnet-land use patterns and satellite imagery
This repository is based on [ControlNet](https://github.com/lllyasviel/ControlNet), which provided the foundation for our implementation.It focuses on using satellite imagery and land use pattern data to train a model with unique Geo Loss and Consistency Loss components. The goal is to generate images that are highly consistent with both geographical and structural features. This tool is designed for the automated processing and generation of large-scale satellite imagery.
For the training process, the output for consistency loss will come with other losses
<img width="1034" height="117" alt="Screenshot 2025-08-21 at 5 55 52 PM" src="https://github.com/user-attachments/assets/adfa33f3-6c14-422c-a132-dc7b629fd7c6" />

The Geo Loss is written in -ldm-models-diffusion-ddpm

mapbox api key and openstreetmap needed for data

model ckpt(7-8 GB) needed for experiments 

Trained with A100 GPU

##Preparation
Before you begin, please ensure you have completed the following steps:

1. Python Environment and Library Installation
This project requires a specific set of libraries to run. Use the following command to install all the necessary dependencies at once:

pip install gradio==3.16.2 \
    albumentations==1.3.0 \
    opencv-contrib-python \
    osmnx==1.3.0 \
    imageio==2.9.0 \
    imageio-ffmpeg==0.4.2 \
    pytorch-lightning==1.5.0 \
    omegaconf==2.1.1 \
    test-tube>=0.7.5 \
    streamlit==1.12.1 \
    einops \
    transformers \
    webdataset==0.2.5 \
    kornia==0.6 \
    open_clip_torch==2.0.2 \
    invisible-watermark>=0.1.5 \
    streamlit-drawable-canvas==0.8.0 \
    torchmetrics==0.6.0 \
    timm==1.0.15 \
    addict==2.4.0 \
    yapf==0.32.0 \
    prettytable==3.6.0 \
    safetensors \
    basicsr==1.4.2 \
    cmake \
    lit


2. Model and Data File Setup
Please ensure your project folder structure matches the relative paths defined in the scripts. This project requires a model checkpoint file (.ckpt) of 7-8 GB to run the experiments.

Additionally, since this project involves geospatial data, you will need the following:

Mapbox API Key

OpenStreetMap data

The Geo Loss implementation is located in the -ldm-models-diffusion-ddpm module.

Project Workflow
The core workflow of this project includes data preparation, training, and generation.

Step 1: Navigate to the Project Directory
First, use the cd command in your terminal to navigate to the project's root directory. 

cd dualcontrolnet3d

Step 2: Data Preparation
See data-download notebook. Make sure you have the necessary scripts to acquire and preprocess data from Mapbox and OpenStreetMap.

Then use clean_data_with_threshold.py to clean the data.

Step 3: Model Training
See first few blocks in train-postprocess notebook. Training requires an A100 GPU for optimal performance. During training, the model is optimized by combining Geo Loss and Consistency Loss.

Step 4: Image Generation
Use the generate_images.py script to batch-generate images. The script will automatically load the model, process the input images, and generate outputs based on the prompts in your CSV file.

python generate_images.py

The generated images will be automatically saved in the written folder.

Script Configuration Parameters
You can customize your generation tasks by editing the main() function in the image_generation_complete.py file. The primary parameters you can modify are:

test_folder: The path to the folder containing input images.

prompt_csv: The path to the CSV file containing prompts and metadata.

output_dir: The output directory where generated images will be saved.

a_prompt: An additional positive prompt to improve image quality.

n_prompt: A negative prompt to exclude unwanted elements.

num_samples: The number of images to generate per run.

image_resolution: The resolution of the output images.

ddim_steps: The number of DDIM diffusion steps.

scale: The strength of unconditional guidance.

resume: If set to True, the script will skip already generated images, allowing for a seamless restart.

