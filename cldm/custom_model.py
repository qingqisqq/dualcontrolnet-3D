import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from cldm.model import create_model
from cldm.cldm import ControlLDM
from ldm.util import instantiate_from_config

class DualControlLDM(ControlLDM):
    def __init__(self, config_path):
        # Load the configuration
        config = OmegaConf.load(config_path)
        model_config = config.model.params
        
        # Call the parent class constructor with the required arguments
        super().__init__(
            control_stage_config=model_config.control_stage_config,
            control_key=model_config.control_key,
            only_mid_control=model_config.only_mid_control if hasattr(model_config, "only_mid_control") else False,
            first_stage_config=model_config.first_stage_config,
            cond_stage_config=model_config.cond_stage_config,
            **{k: v for k, v in model_config.items() 
               if k not in ['first_stage_config', 'cond_stage_config', 'control_stage_config', 
                           'control_key', 'only_mid_control']}
        )
        
        # Create the second control model
        base_model = create_model(config_path)
        self.control_model_remote = self.control_model  # Use the one from parent
        self.control_model_landuse = base_model.control_model  # Create a new one
        self.consistency_weight = 0.1

    def load_control_net_weights(self, paths):
        """加载两个 ControlNet 的权重"""
        remote_state_dict = torch.load(paths[0], map_location="cpu")
        landuse_state_dict = torch.load(paths[1], map_location="cpu")
        self.control_model_remote.load_state_dict(remote_state_dict)
        self.control_model_landuse.load_state_dict(landuse_state_dict)
        print(f"Loaded ControlNet weights for remote and landuse models.")

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        """
        使用两个 ControlNet 模型分别生成预测，并计算一致性损失
        """
        control_images = cond["c_concat"][0]  # 控制图像
        remote_images = cond["remote_images"]  # 遥感目标图像
        landuse_images = cond["landuse_images"]  # 土地利用目标图像

        # 使用两个 ControlNet 模型生成预测
        control_remote = self.control_model_remote(
            x_noisy, hint=control_images, timesteps=t, context=cond["c_crossattn"][0]
        )
        control_landuse = self.control_model_landuse(
            x_noisy, hint=control_images, timesteps=t, context=cond["c_crossattn"][0]
        )

        # 计算每个任务的损失
        loss_remote = F.mse_loss(control_remote, remote_images)
        loss_landuse = F.mse_loss(control_landuse, landuse_images)

        # 计算潜在空间一致性损失
        latent_remote = self.first_stage_model.encode(remote_images).sample()
        latent_landuse = self.first_stage_model.encode(landuse_images).sample()
        consistency_loss = F.mse_loss(latent_remote, latent_landuse)

        # 总损失
        total_loss = loss_remote + loss_landuse + self.consistency_weight * consistency_loss
        return total_loss

    def training_step(self, batch, batch_idx):
        """
        重写 training_step 方法
        """
        x_noisy, cond = self.get_input(batch, self.first_stage_key)
        t = torch.randint(0, self.num_timesteps, (x_noisy.size(0),), device=self.device).long()
        loss = self.apply_model(x_noisy, t, cond)
        self.log("train_loss", loss)
        return loss