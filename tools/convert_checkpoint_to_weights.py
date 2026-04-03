import torch as th
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

checkpoint_path = BASE_DIR / "checkpoints" / "kitti_finetune_checkpoint_3e-4_epoch_50.pt"
output_weights_path = BASE_DIR / "checkpoints" / "kitti_finetune_model_weights_3e-4_epoch_50.pt"

checkpoint = th.load(checkpoint_path, map_location="cpu")
model_weights = checkpoint["model_state_dict"]

th.save(model_weights, output_weights_path)

print(f"Saved pure model weights to: {output_weights_path}")