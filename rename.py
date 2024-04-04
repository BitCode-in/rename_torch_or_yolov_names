import torch
from ultralytics import YOLO

model = torch.load("best.pt", map_location="cpu")

modelc = (model.get("ema") or model["model"]).to("cpu").float()
modelc.names[0] = "New_name"

torch.save(model, "save_best.pt")

model = YOLO(r"save_best.pt")
print(model.names)
