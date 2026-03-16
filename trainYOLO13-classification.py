from ultralytics import YOLO
import numpy as np

model = YOLO("/Users/maisycook/Documents/Plant Disease Detection/~/project/Models/yolo11n-cls.pt")

model.train(
    data="/Users/maisycook/Documents/Plant Disease Detection/degradedclassificationDataset",
    imgsz=224,
    batch=32,
    epochs=30,
    device="cpu",
    patience=20,
    workers=4,
    cache="ram",
    amp=True,
    augment=False
)