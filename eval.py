import torch
from model import *
import h5py
model = PSDStructuralProbe(768, 128)
state_dict=torch.load("depth probe.pt", weights_only=True)
model.load_state_dict(state_dict,strict=True)

import torch

def spearman_corr(x, y):
    """
    x, y: 形状相同的 1D tensor
    返回 Spearman 秩相关系数（标量）
    """
    # 1. 计算排名（argsort 两次得到排名）
    x_rank = torch.argsort(torch.argsort(x)).float() + 1
    y_rank = torch.argsort(torch.argsort(y)).float() + 1
    
    # 2. 中心化
    x_centered = x_rank - x_rank.mean()
    y_centered = y_rank - y_rank.mean()
    
    # 3. Pearson 相关系数
    numerator = (x_centered * y_centered).sum()
    denominator = torch.sqrt((x_centered**2).sum() * (y_centered**2).sum())
    
    return numerator / denominator

with h5py.File("dependency_test_data.h5") as f:
    print(f"Length: {len(f["word_depth"]["X"])}")
    X = f["word_depth"]["X"][:,5,:]
    X = torch.Tensor(X)
    depth = torch.Tensor(f["word_depth"]["depth"][:])
    pred = model(X)
    pred_depth = torch.sum(pred*pred, dim=1)
    print("词深度 Spearman 相关系数:",spearman_corr(depth, pred_depth))

    state_dict=torch.load("distance probe.pt", weights_only=True)
    model.load_state_dict(state_dict,strict=True)
    X_diff = f["word_distance"]["X_diff"][:, 5, :]
    X_diff = torch.Tensor(X_diff)
    distance = torch.Tensor(f["word_distance"]['distance'][:])
    pred = model(X_diff)
    pred_distance = torch.sum(pred*pred, dim=1)
    print("词距离 Spearman 相关系数:", spearman_corr(distance, pred_distance))