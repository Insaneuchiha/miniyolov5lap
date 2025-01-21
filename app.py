import os
from flask import Flask, render_template, send_file, jsonify
import torch
from PIL import Image

app = Flask(__name__)

# Load the YOLO model (v5)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load the pre-trained YOLOv5 small model

# Define function to get the latest image
def get_latest_image(folder_path):
    # Get all files in the folder
    files = os.listdir(folder_path)
    # Filter only image files (e.g., .jpg, .jpeg)
    images = [f for f in files if f.endswith(('.jpg', '.jpeg', '.png'))]
    # Get the full path of the latest image based on creation time
    latest_image = max(images, key=lambda f: os.path.getctime(os.path.join(folder_path, f)))
    return os.path.join(folder_path, latest_image)

# Route to display the count of cars and bikes
@app.route('/')
def index():
    # Get the latest image from the 'images' folder inside YOLOv5 directory
    folder_path = 'D:/YOLO/yolov5/images'  # Adjust this path as necessary
    latest_image_path = get_latest_image(folder_path)

    # Process the latest image using YOLOv5
    img = Image.open(latest_image_path)
    results = model(img)  # Perform object detection

    # Filter the results for cars and bikes
    cars = 0
    bikes = 0
    for pred in results.xywh[0]:  # Loop through the predictions
        if pred[5] == 2:  # Car class ID (YOLOv5 - class 2 is car)
            cars += 1
        elif pred[5] == 3:  # Bike class ID (YOLOv5 - class 3 is motorcycle)
            bikes += 1

    # Pass the counts and the image path to the HTML template
    return render_template('index.html', cars=cars, bikes=bikes, image_path=latest_image_path)

# Route to serve the latest image
@app.route('/latest_image')
def latest_image():
    # Get the latest image path from the same function
    folder_path = 'D:/YOLO/yolov5/images'
    latest_image_path = get_latest_image(folder_path)

    # Serve the latest image as static content
    return send_file(latest_image_path, mimetype='image/jpeg')

# Route to fetch the latest counts as JSON
@app.route('/latest_data')
def latest_data():
    # Get the latest image path from the same function
    folder_path = 'D:/YOLO/yolov5/images'
    latest_image_path = get_latest_image(folder_path)

    # Process the latest image using YOLOv5
    img = Image.open(latest_image_path)
    results = model(img)  # Perform object detection

    # Filter the results for cars and bikes
    cars = 0
    bikes = 0
    for pred in results.xywh[0]:  # Loop through the predictions
        if pred[5] == 2:  # Car class ID (YOLOv5 - class 2 is car)
            cars += 1
        elif pred[5] == 3:  # Bike class ID (YOLOv5 - class 3 is motorcycle)
            bikes += 1

    # Return the counts as JSON
    return jsonify({'cars': cars, 'bikes': bikes})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
