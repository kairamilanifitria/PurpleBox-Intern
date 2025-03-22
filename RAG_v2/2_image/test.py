import torch
print("CUDA Available:", torch.cuda.is_available())  # Should print True
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")
