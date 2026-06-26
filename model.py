import torch
import torch.nn as nn

class PSDStructuralProbe(nn.Module):
    def __init__(self, embedding_dim, rank_d):
        super().__init__()
        self.B = nn.Linear(embedding_dim, rank_d, bias=False)

    def forward(self, X):
        #X:(batch_size, embedding_dim)
        #这里返回的是原文的Bh，最终的点积形式在loss里算
        return self.B(X)
    
class NonPSDStructuralProbe(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.A = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, X):
        #X:(batch_size, embedding_dim)
        temp = self.A(X)
        #这里直接返回双线性函数值了
        return torch.sum(temp*X, dim=1)