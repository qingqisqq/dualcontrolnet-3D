# dualcontrolnet-land use patterns and satellite imagery
For the training process, the output for consistency loss will come with other losses
<img width="1034" height="117" alt="Screenshot 2025-08-21 at 5 55 52 PM" src="https://github.com/user-attachments/assets/adfa33f3-6c14-422c-a132-dc7b629fd7c6" />

The Geo Loss is written in -ldm-models-diffusion-ddpm

This repository is based on [ControlNet](https://github.com/lllyasviel/ControlNet), which provided the foundation for our implementation

mapbox api key and openstreetmap needed for data

model ckpt(7-8 GB) needed for experiments 

Trained with A100 GPU

1. start with environment set up

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
    
2.use the command cd dualcontrolnet3d to locate the project directory

3. data
4.
5. train
4.generate
5. postprocess


