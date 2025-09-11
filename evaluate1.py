import numpy as np
import os
import time
import torch
from torchvision import models, transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import random
import shutil

class ImageDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.files = [f for f in os.listdir(path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.path, self.files[idx])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

def extract_features(dataloader, model, device):
    model.eval()
    features = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            feat = model(batch)
            features.append(feat.cpu().numpy())
    
    features = np.concatenate(features, axis=0)
    return features

def calculate_fid(real_features, gen_features):
    # 计算均值
    mu1 = np.mean(real_features, axis=0)
    mu2 = np.mean(gen_features, axis=0)
    
    # 计算协方差矩阵
    sigma1 = np.cov(real_features, rowvar=False)
    sigma2 = np.cov(gen_features, rowvar=False)
    
    # 计算均值差的平方
    diff = mu1 - mu2
    
    # 添加一个小的正则化项到对角线
    eps = 1e-6
    sigma1 = sigma1 + np.eye(sigma1.shape[0]) * eps
    sigma2 = sigma2 + np.eye(sigma2.shape[0]) * eps
    
    # 计算协方差矩阵的平方根
    covmean_sq = sigma1.dot(sigma2)
    
    # 确保矩阵是正定的
    if np.iscomplexobj(covmean_sq):
        print("警告：协方差矩阵包含复数，使用实部")
        covmean_sq = covmean_sq.real
    
    # 计算矩阵平方根的迹
    # 使用特征值分解来计算
    eigvals = np.linalg.eigvals(covmean_sq)
    eigvals = np.maximum(eigvals, 0)  # 确保特征值为正
    covmean_trace = np.sum(np.sqrt(eigvals))
    
    # 计算FID
    fid = np.sum(diff**2) + np.trace(sigma1) + np.trace(sigma2) - 2 * covmean_trace
    
    return fid

# 主函数
def main():
    # 使用您指定的文件夹路径
    real_merged_dir = "./merged6_real_images"
    gen_merged_dir = "./merged66_generated_images"
    
    # 创建临时目录进行平衡处理
    temp_real_dir = "./temp_real_images"
    temp_gen_dir = "./temp_gen_images"
    
    try:
        # 创建临时目录
        os.makedirs(temp_real_dir, exist_ok=True)
        os.makedirs(temp_gen_dir, exist_ok=True)
        
        # 获取两个文件夹中的图片数量
        real_images = [f for f in os.listdir(real_merged_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        gen_images = [f for f in os.listdir(gen_merged_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        # 确定最小数量
        min_count = min(len(real_images), len(gen_images))
        
        # 随机选择相同数量的图片
        selected_real = random.sample(real_images, min_count)
        selected_gen = random.sample(gen_images, min_count)
        
        # 复制到临时目录
        for img in selected_real:
            shutil.copy(os.path.join(real_merged_dir, img), os.path.join(temp_real_dir, img))
        
        for img in selected_gen:
            shutil.copy(os.path.join(gen_merged_dir, img), os.path.join(temp_gen_dir, img))
        
        print(f"已平衡两个文件夹的图片数量至 {min_count} 张")
        
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载预训练的Inception模型
        model = models.inception_v3(pretrained=True, transform_input=False)
        # 移除最后的分类层
        model.fc = torch.nn.Identity()
        model = model.to(device)
        
        # 设置图像转换
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # 创建数据集和数据加载器
        real_dataset = ImageDataset(temp_real_dir, transform)
        gen_dataset = ImageDataset(temp_gen_dir, transform)
        
        real_loader = DataLoader(real_dataset, batch_size=32, shuffle=False, num_workers=4)
        gen_loader = DataLoader(gen_dataset, batch_size=32, shuffle=False, num_workers=4)
        
        # 记录开始时间
        start_time = time.time()
        
        # 提取特征
        print("提取真实图像特征...")
        real_features = extract_features(real_loader, model, device)
        print("提取生成图像特征...")
        gen_features = extract_features(gen_loader, model, device)
        
        # 计算FID
        print("计算FID值...")
        fid_value = calculate_fid(real_features, gen_features)
        
        # 计算耗时
        elapsed_time = time.time() - start_time
        
        print(f"FID值: {fid_value}")
        print(f"计算耗时: {elapsed_time:.2f} 秒")
    
    finally:
        # 清理临时目录
        if os.path.exists(temp_real_dir):
            shutil.rmtree(temp_real_dir)
        if os.path.exists(temp_gen_dir):
            shutil.rmtree(temp_gen_dir)

if __name__ == "__main__":
    main()
