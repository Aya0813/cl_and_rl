import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("Number of GPUs:", torch.cuda.device_count())
    print("Current GPU:", torch.cuda.current_device())
    print("GPU name:", torch.cuda.get_device_name(0))

    x = torch.randn(1000, 1000).cuda()
    y = x @ x
    print("Tensor test passed. y shape:", y.shape)
else:
    print("No CUDA GPU detected by PyTorch.")