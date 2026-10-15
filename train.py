import torch
from torch.utils.data import DataLoader
from datasets import DepthDataSet, DistanceDataSet
from model import *
from loss import *
import argparse
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="trainer")
    parser.add_argument("-d", '--dataset', choices=['depth', 'distance'], default="depth", help='探针数据集类型')
    parser.add_argument("-m", '--model', choices=['psd', 'nonpsd'], default='psd', help='模型')
    parser.add_argument('-e', '--num_epochs', type=int, default=10, help="训练轮数")
    parser.add_argument("--device", choices=['cpu', 'mps', 'cuda'], default='cpu', help='训练设备')
    args = parser.parse_args()
    if args.dataset == 'depth':
        dataset = DepthDataSet("dependency_train_data.h5")
    elif args.dataset == 'distance':
        dataset = DistanceDataSet("dependency_train_data.h5")

    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False
    )
    if args.device == "mps":
        device = torch.device("mps")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"当前设备:{device}")

    if args.model == 'psd':
        model = PSDStructuralProbe(768, 128)
        criterion = PSDProbeLoss()
    elif args.model == "nonpsd":
        model = NonPSDStructuralProbe(768)
        criterion = NonPSDProbeLoss()
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), 1e-3, betas=(0.9, 0.999), eps=1e-8)
    for epoch in tqdm(range(args.num_epochs), desc='Epoch: '):
        batch_bar = tqdm(loader, desc='Batch: ')
        total_loss = 0
        for batch_x, batch_y in batch_bar:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            batch_y_pred = model(batch_x)
            loss = criterion(batch_y_pred, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_bar.set_postfix({"Batch Loss: ": f"{loss.item():.4f}"})
        avg_loss = total_loss / len(loader)
        print("Epoch Average Loss: ", avg_loss)
    if torch.save(model.state_dict(), f"{args.dataset} probe.pt"):
        print(f"模型已保存至 {args.dataset} probe.pt")