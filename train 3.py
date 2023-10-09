import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if torch.cuda.is_available():
    print("CUDA (GPU) is available.")
else:
    print("CUDA (GPU) is not available.")
