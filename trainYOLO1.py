from ultralytics import YOLO

model = YOLO("/Users/maisycook/Documents/Plant Disease Detection/~/project/Models/yolo26n.pt")

model.train(data="/Users/maisycook/Documents/Plant Disease Detection/plantDisease_dataset.yaml",  imgsz = 640, batch = 16, epochs = 100, workers = 0, cache = True, device = 0)


