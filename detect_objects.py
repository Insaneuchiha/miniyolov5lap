import torch
from PIL import Image

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load pretrained YOLOv5 model

# Load your image
# img = Image.open('D:\YOLO\yolov5\images\test2.jpg')  # Replace with the path to your image
img = Image.open(r'D:\YOLO\yolov5\images\test2.jpg')


# Run inference on the image
results = model(img)

# Get detected object labels (e.g., car=2, motorbike=3 in COCO dataset)
labels = results.xywh[0][:, -1].numpy()  # Get label IDs

# Count cars and bikes
car_count = sum(label == 2 for label in labels)  # Label 2 is for cars
bike_count = sum(label == 3 for label in labels)  # Label 3 is for bikes

print(f"Cars: {car_count}, Bikes: {bike_count}")

