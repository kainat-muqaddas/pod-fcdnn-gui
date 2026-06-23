from pathlib import Path
import torch

path = Path(
    r"D:\project\pod-fcdnn-gui-main\checkpoints\cavity_checkpoint.pt"
)

print("Exists:", path.exists())
print("Path:", path)

ckpt = torch.load(path, map_location="cpu")

print("Checkpoint loaded successfully")
print("Keys:", ckpt.keys())
