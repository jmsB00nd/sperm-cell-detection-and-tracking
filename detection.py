import os
import cv2
import matplotlib.pyplot as plt
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.predict import get_sliced_prediction
import torch

# Define the model path and detection parameters
yolov5_model_path = "model/best.pt"

# Set the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the detection model
detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov5",
    model_path=yolov5_model_path,
    confidence_threshold=0.3,
    device=device,
)

# Perform object detection on an image by dividing it into smaller overlapping slices
image_path = '/content/datasets/internship-object-detection-2/valid/images/tile_1_0_4_frames_16_jpg.rf.ac4838cee022483e0873a13bf08fb98d.jpg'
result = get_sliced_prediction(
    image_path,
    detection_model,
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2
)

# Export and display the results
export_dir = "results/"
if not os.path.exists(export_dir):
    os.makedirs(export_dir)
result.export_visuals(export_dir=export_dir)

# Load the exported image and display it using OpenCV or matplotlib
visual_image_path = os.path.join(export_dir, "prediction_visual.png")
visual_image = cv2.imread(visual_image_path)

# Display the image using matplotlib
plt.imshow(cv2.cvtColor(visual_image, cv2.COLOR_BGR2RGB))
plt.axis('off')  # Turn off axis
plt.show()
