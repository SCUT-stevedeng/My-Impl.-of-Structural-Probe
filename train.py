import torch
from torch.utils.data import DataLoader
from datasets import DepthDataSet, DistanceDataSet
from model import *
from loss import *

dataset = DepthDataSet("dependency_train_data.h5")
loader = DataLoader(
    dataset,
    batch_size=512,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
    persistent_workers=False
)

model = PSDStructuralProbe(768, 128)
criterion = PSDProbeLoss()
optimizer = torch.optim.Adam(model.parameters(), 1e-3)
total_loss = 0
for batch_x, batch_y in loader:
    optimizer.zero_grad()
    batch_y_pred = model(batch_x)
    loss = criterion(batch_y_pred, batch_y)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
    print(f"batch loss: {loss.item()}")
avg_loss = total_loss / len(loader)
print("Average Loss: ", avg_loss)