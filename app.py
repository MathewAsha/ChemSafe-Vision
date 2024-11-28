from flask import Flask, request, render_template, jsonify
import cv2
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from ultralytics import YOLO  # Import YOLOv8 from ultralytics

app = Flask(__name__)

# Load the fine-tuned YOLOv8 model
model = YOLO('best.pt')  # Replace with your YOLOv8 weights file

# Function to process the image and get bounding boxes
def detect_hazards(image):
    # Convert the image to RGB format if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Convert the image to OpenCV format
    img_array = np.array(image)
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Perform inference using the YOLOv8 model
    results = model.predict(img)

    # Get bounding box data from results
    bbox_data = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            label = model.names[int(box.cls[0])]
            bbox_data.append({
                'xmin': x1, 'ymin': y1, 'xmax': x2, 'ymax': y2,
                'confidence': conf, 'name': label
            })
            # Draw bounding boxes on the image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{label} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert the image back to base64 for displaying in HTML
    _, buffer = cv2.imencode('.jpg', img)
    encoded_image = base64.b64encode(buffer).decode()

    return bbox_data, encoded_image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})

    # Get the image file
    image_file = request.files['image']
    image = Image.open(image_file.stream)

    # Detect hazards and get bounding boxes
    bbox_data, processed_image = detect_hazards(image)

    # Return the bounding boxes and image to the user
    return render_template('result.html', bboxes=bbox_data, image_data=processed_image)

if __name__ == "__main__":
    app.run(debug=True)
