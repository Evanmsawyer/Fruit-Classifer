import torch 
print(torch.cuda.is_available())  # Should print True if CUDA is available
print(torch.version.cuda)         # Should show your CUDA version