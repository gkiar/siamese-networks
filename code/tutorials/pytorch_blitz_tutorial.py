#!/usr/bin/env python

import torch

# Creating tensors
# x = torch.zeros(5, 3, dtype=torch.long)
# x = torch.tensor([5.5, 3])
# x = torch.randn_like(x, dtype=torch.float)
x = torch.ones(5, 3, dtype=torch.float)
y = torch.rand(5, 3)

# Adding tensors
# x+y
# torch.add(x, y)
# y.add_(x)
result = torch.empty(5, 3)
torch.add(x, y, out=result)

x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)
print(x.size(), y.size(), z.size())

# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!
