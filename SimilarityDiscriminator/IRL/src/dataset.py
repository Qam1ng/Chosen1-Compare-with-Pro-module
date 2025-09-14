# src/dataset.py

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class TrajectoryDataset(Dataset):
    """
    用于加载轨迹片段及其标签的数据集。
    
    从一个metadata CSV文件读取数据，该文件包含轨迹文件路径和标签。
    对于每个样本，它会从轨迹文件中随机采样一个固定长度的片段。
    """
    def __init__(self, metadata_file: str, data_dir: str, sequence_length: int):
        """
        Args:
            metadata_file (str): metadata CSV文件的路径。
            data_dir (str): 存放处理后轨迹数据(.npy)的目录。
            sequence_length (int): 每个样本的固定序列长度。
        """
        self.metadata = pd.read_csv(metadata_file)
        self.data_dir = data_dir
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.metadata.iloc[idx]
        file_path = f"{self.data_dir}/{row['filepath']}"
        label = row['label']

        # 加载完整的轨迹数据
        full_trajectory = np.load(file_path, allow_pickle=True).astype(np.float32)

        # 从完整轨迹中随机采样一个固定长度的片段
        start_idx = 0
        trajectory_len = full_trajectory.shape[0]
        if trajectory_len > self.sequence_length:
            start_idx = np.random.randint(0, trajectory_len - self.sequence_length)
        
        end_idx = start_idx + self.sequence_length
        
        sampled_trajectory = full_trajectory[start_idx:end_idx]

        # 如果原始轨迹比期望的短，进行填充 (padding)
        if sampled_trajectory.shape[0] < self.sequence_length:
            padding_needed = self.sequence_length - sampled_trajectory.shape[0]
            # 使用0向量进行填充
            padding = np.zeros((padding_needed, sampled_trajectory.shape[1]), dtype=np.float32)
            sampled_trajectory = np.vstack([sampled_trajectory, padding])

        # 转换为PyTorch Tensors
        trajectory_tensor = torch.from_numpy(sampled_trajectory)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return trajectory_tensor, label_tensor