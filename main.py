from detector import PlayerDetector
from tracker import PlayerTracker
import cv2
import os

# Paths
video_path = "input/15sec_input_720p.mp4"
output_path = "output/reid_result.mp4"

# Initialize detector and tracker
detector = PlayerDetector(weights_path="yolov11\yolov11_custom.pt")
tracker = PlayerTracker()

# Open video
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output video writer
os.makedirs("output", exist_ok=True)
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

frame_id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = detector.detect(frame)
    tracked_frame = tracker.update(frame, detections, frame_id)
    out.write(tracked_frame)

    frame_id += 1

cap.release()
out.release()
print("✅ Re-Identification Completed: Saved to", output_path)
