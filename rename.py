import torch
from ultralytics import YOLO

model = torch.load("best.pt", map_location="cpu")

model["model"].names[0] = "New name"

torch.save(model, "save_best.pt")

model = YOLO(r"save_best.pt")
print(model.names)
