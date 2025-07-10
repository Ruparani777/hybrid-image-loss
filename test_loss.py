import torch
from loss import HybridLoss

if __name__ == "__main__":
    loss_fn = HybridLoss()
    input_tensor = torch.randn(1, 3, 256, 256)
    target_tensor = torch.randn(1, 3, 256, 256)
    loss = loss_fn(input_tensor, target_tensor)
    print("Loss:", loss.item())
