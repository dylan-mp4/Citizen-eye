import torch

# check if a compatible GPU is available
if torch.cuda.is_available():
    # get the index of the current GPU device being used
    device_idx = torch.cuda.current_device()
    print(f"GPU Device Detected: Using device {device_idx}")
else:
    print("GPU is not available/undetected, attempt to use CPU, this will degrade performance")

# create a tensor on CPU
x = torch.tensor([1, 2, 3])

# move the tensor to GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
    x = x.to(device)
    print("GPU is able to be used for tensor operations")
else:
    print("GPU is not available/undetected")