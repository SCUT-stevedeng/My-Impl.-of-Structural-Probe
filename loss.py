import torch
import torch.nn as nn

class PSDProbeLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, projection, dist_or_norm):
        n = projection.shape[0]
        dist_or_norm_pred = torch.sum(projection*projection, dim=1)
        cost = torch.abs(dist_or_norm_pred - dist_or_norm)
        if self.reduction == "mean":
            loss = torch.sum(cost) / n
        elif self.reduction == "sum":
            loss = torch.sum(cost)
        return loss
    
class NonPSDProbeLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, metrics, dist_or_norm):
        n = len(metrics)
        cost = torch.abs(metrics - dist_or_norm)
        if self.reduction == "mean":
            loss = torch.sum(cost) / n
        elif self.reduction == "sum":
            loss = torch.sum(cost)
        return loss
    