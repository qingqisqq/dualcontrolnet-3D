import os
import torch

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config


def get_state_dict(d):
    return d.get('state_dict', d)


def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict


def create_model(config_path):
    config = OmegaConf.load(config_path)
    
    # 创建模型前先提取 land_use_decoder_config
    land_use_decoder_config = None
    if hasattr(config.model, 'land_use_decoder_config'):
        land_use_decoder_config = config.model.land_use_decoder_config
    
    # 实例化模型
    model = instantiate_from_config(config.model).cpu()
    
    # 如果有 land_use_decoder_config，创建并设置解码器
    if land_use_decoder_config is not None:
        land_use_decoder = instantiate_from_config(land_use_decoder_config).cpu()
        model.land_use_decoder = land_use_decoder
    
    print(f'Loaded model config from [{config_path}]')
    return model
