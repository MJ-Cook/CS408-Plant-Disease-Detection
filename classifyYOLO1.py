from ultralytics import YOLO

model = YOLO('/Users/maisycook/Documents/Plant Disease Detection/runs/classify/train-server2/weights/best.pt')
image_path = '/Users/maisycook/Documents/Plant Disease Detection/testing images/powderymildew-cherry.jpg'

results = model(image_path)

for r in results:
    probs = r.probs
    names = r.names

    top5 = probs.top5
    top5conf = probs.top5conf

    print("\n5 Highest Confidence Diagnoses:")
    for i in range(len(top5)):
        class_id = top5[i]
        confidence = float(top5conf[i])
        print(f"{names[class_id]}: {confidence:.4f}")