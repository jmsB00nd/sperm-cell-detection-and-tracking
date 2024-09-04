import os
import cv2
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.predict import get_prediction
from sahi.predict import get_sliced_prediction

# Define the model path and detection parameters
yolov5_model_path = "model/best.pt"

# Load the detection model
detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov5",
    model_path=yolov5_model_path,
    confidence_threshold=0.3,
    device="cpu",
)

# Perform object detection on the input image
image_path = '/home/jmsbnd/Documents/AI/sperm-cell-detection-and-tracking/data/images/tile_0_0_1_frames_4.jpg'
image = read_image(image_path)


#result = get_prediction(image, detection_model)

# Perform object detection on an image by dividing it into smaller overlapping slices.
# This is useful for large images that might not fit into memory or when higher resolution detection is needed.
result = get_sliced_prediction(
    '/content/datasets/internship-object-detection-2/valid/images/tile_1_0_4_frames_16_jpg.rf.ac4838cee022483e0873a13bf08fb98d.jpg',
    detection_model,
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2)
# Export and display the results
export_dir = "results/"
result.export_visuals(export_dir=export_dir)

# Load the exported image and display it using OpenCV
visual_image_path = os.path.join(export_dir, "prediction_visual.png")
visual_image = cv2.imread(visual_image_path)

# Display the image
cv2.imshow('Detection Result', visual_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
