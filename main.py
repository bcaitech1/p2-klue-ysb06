import torch

print(f"PyTorch version: [{torch.__version__}]")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"         device: [{device}]")