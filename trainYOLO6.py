from ultralytics import YOLO
import numpy as np

model = YOLO("/Users/maisycook/Documents/Plant Disease Detection/~/project/Models/yolo26n.pt")

model.train(data="/Users/maisycook/Documents/Plant Disease Detection/plantDisease_dataset.yaml", imgsz=416, batch=8, epochs=80, amp=True, workers=2, patience=20, cache=True)


