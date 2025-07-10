# Import the YOLO class from the ultralytics package
from ultralytics import YOLO

class PlayerDetector:
    def __init__(self, weights_path):
        """
        Initializes the PlayerDetector with a pre-trained YOLO model.

        Parameters:
        weights_path (str): Path to the trained YOLO weights (.pt file)
        """
        # Load the YOLO model using the provided weights
        self.model = YOLO(weights_path)

    def detect(self, frame):
        """
        Runs object detection on the input frame and extracts bounding boxes.

        Parameters:
        frame (np.ndarray or str): The input image/frame or video path for detection

        Returns:
        boxes (list): A list of detected bounding boxes with confidence and class info.
                      Each box is [x1, y1, x2, y2, confidence, class_id]
        """
        # Perform prediction using the YOLO model
        # - source: the input frame
        # - conf=0.4: only consider detections with confidence > 0.4
        # - verbose=False: disables detailed logging
        results = self.model.predict(source=frame, conf=0.4, verbose=False)

        boxes = []  # Initialize list to store detected bounding boxes

        # Iterate through prediction results
        for r in results:
            # For each detected box, convert coordinates and info into a usable format
            for box in r.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = box  # Extract box corners, confidence, and class
                # Append the processed box to the list (converted to appropriate types)
                boxes.append([int(x1), int(y1), int(x2), int(y2), float(conf), int(cls)])

        return boxes  # Return the list of detected boxes
