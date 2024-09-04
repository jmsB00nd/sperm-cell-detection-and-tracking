import cv2
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import csv
from datetime import datetime

# Load YOLOv5 model with custom weights
model = torch.hub.load('ultralytics/yolov5', 'custom', path='model/best.pt')
model.eval()

# Initialize DeepSORT
deepsort = DeepSort(max_age=30, n_init=3, nn_budget=100)

# Open video file
cap = cv2.VideoCapture('data/videos/sperm.mp4')

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create VideoWriter object
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_video = cv2.VideoWriter(f'output_video_{timestamp}.mp4', 
                               cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Prepare CSV file for tracking data
csv_filename = f'tracking_data_{timestamp}.csv'
csv_file = open(csv_filename, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Frame', 'Track_ID', 'X', 'Y', 'Width', 'Height', 'Confidence'])

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Perform YOLOv5 inference
    with torch.amp.autocast('cuda'):
        results = model(frame)

    # Extract bounding boxes, scores, and classes from YOLOv5 results
    detections = results.pandas().xyxy[0]
    
    # Prepare detections for DeepSORT
    detection_list = []
    for _, detection in detections.iterrows():
        x1, y1, x2, y2 = detection[['xmin', 'ymin', 'xmax', 'ymax']]
        conf = detection['confidence']
        cls = detection['class']
        detection_list.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

    # Update DeepSORT tracker
    if detection_list:
        tracks = deepsort.update_tracks(detection_list, frame=frame)

        # Draw tracking results on the frame and save data
        for track in tracks:
            if not track.is_confirmed():
                continue
            bbox = track.to_tlbr()  # Get bounding box in top-left bottom-right format
            track_id = track.track_id
            
            # Draw on frame
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {track_id}', (int(bbox[0]), int(bbox[1]) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Save to CSV
            csv_writer.writerow([frame_count, track_id, bbox[0], bbox[1], 
                                 bbox[2] - bbox[0], bbox[3] - bbox[1], track.get_det_conf()])

    # Write the frame to output video
    output_video.write(frame)

    
    cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
output_video.release()
cv2.destroyAllWindows()
csv_file.close()

print(f"Tracking complete. Results saved to {csv_filename} and output_video_{timestamp}.mp4")