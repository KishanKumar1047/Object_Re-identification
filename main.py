from detector import PlayerDetector        # Custom class to detect players using YOLOv11
from tracker import PlayerTracker          # Custom class to track detected players using appearance similarity
import cv2
import os

# --- Input and Output Paths ---
video_path = "input/15sec_input_720p.mp4"              # Path to input video
output_path = "output/reid_result.mp4"                 # Path where the output video with tracking will be saved

# --- Initialize YOLO Detector and Appearance-Based Tracker ---
detector = PlayerDetector(weights_path="yolov11\\yolov11_custom.pt")  # Load trained YOLOv11 model
tracker = PlayerTracker()                                             # Initialize the tracker

# --- Open the Input Video ---
cap = cv2.VideoCapture(video_path)       # Open video file for reading
fps = int(cap.get(cv2.CAP_PROP_FPS))     # Get frames per second of the video
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))     # Frame width
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))    # Frame height

# --- Setup Video Writer to Save Output ---
os.makedirs("output", exist_ok=True)     # Create output directory if not exists
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

frame_id = 0  # Frame counter for logging or tracking logic

# --- Process Video Frame by Frame ---
while True:
    ret, frame = cap.read()      # Read a frame from the video
    if not ret:                  # Break if no more frames
        break

    # Detect players in the frame
    detections = detector.detect(frame)

    # Track players across frames using ID association
    tracked_frame = tracker.update(frame, detections, frame_id)

    # Write the annotated frame to output video
    out.write(tracked_frame)

    frame_id += 1

# --- Release Resources ---
cap.release()
out.release()

# --- Done ---
print("✅ Re-Identification Completed: Saved to", output_path)
