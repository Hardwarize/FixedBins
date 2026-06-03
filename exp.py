import torch
from depth_estimation.model.model import UDFNet

DEVICE = 'cpu'

model = UDFNet(n_bins=80, max_depth=1.0).to(DEVICE)

dummy_rgb = torch.rand((3,200, 400)).unsqueeze(dim=0)
dummy_depth = torch.rand((2, 100, 200)).unsqueeze(dim=0)

model(dummy_rgb,dummy_depth)


