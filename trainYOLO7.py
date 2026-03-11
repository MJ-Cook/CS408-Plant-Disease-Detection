from ultralytics import YOLO
import numpy as np

model = YOLO("/Users/maisycook/Documents/Plant Disease Detection/~/project/Models/yolo26n.pt")

model.train(data="/Users/maisycook/Documents/Plant Disease Detection/plantDisease_dataset.yaml", imgsz=512, batch=8, epochs=80, amp=False, device="cpu", patience=20, workers=8, cache="ram", rect=True, fraction=0.25)


