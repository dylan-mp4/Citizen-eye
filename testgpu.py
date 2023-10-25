import torch

# check if a compatible GPU is available
if torch.cuda.is_available():
    # get the index of the current GPU device being used
    device_idx = torch.cuda.current_device()
    print(f"Using GPU device {device_idx}")
else:
    print("GPU is not available, using CPU instead")

# create a tensor on CPU
x = torch.tensor([1, 2, 3])

# move the tensor to GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
    x = x.to(device)
    print("Tensor moved to GPU")
else:
    print("Tensor is on CPU")