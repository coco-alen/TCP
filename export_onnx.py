import os
import time

import torch

from TCP.config import GlobalConfig
from TCP.model import TCP

config = GlobalConfig()
model = TCP(config)
print(model)
image = torch.randn(1, 3, 256, 900, device='cuda')
measure = torch.randn(1, 9, device='cuda')
target = torch.randn(1, 2, device='cuda')
model = model.cuda()

torch.onnx.export(model, (image, measure, target), "TCP.onnx", verbose=True, opset_version=12,
                  input_names=['image', 'measure', 'target'],
                  output_names=['output'])