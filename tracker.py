from utils import draw_bbox, cosine_similarity
import numpy as np
import cv2

class PlayerTracker:
    def __init__(self):
        """
        Initializes the player tracker with an empty dictionary of players
        and sets up a dummy feature extractor for appearance matching.
        """
        self.next_id = 0  # Unique ID to assign to new players
        self.players = {}  # Dictionary to store player info: {id: [bbox, feature_vector]}
        
        # Dummy feature extractor using simple resizing and flattening
        self.feature_extractor = lambda crop: cv2.resize(crop, (64, 128)).flatten()

    def update(self, frame, detections, frame_id):
        """
        Updates the tracking state with the latest detections.

        Parameters:
        - frame (np.ndarray): The current video frame.
        - detections (list): List of detected bounding boxes with format 
                             [x1, y1, x2, y2, confidence, class_id].
        - frame_id (int): The index of the current frame (optional usage for logging or analysis).

        Returns:
        - frame (np.ndarray): The frame with bounding boxes and player IDs drawn.
        """
        new_players = {}  # Temp dictionary to store updated tracking info

        for det in detections:
            x1, y1, x2, y2, conf, cls = det  # Extract detection values

            # Crop the detected region from the frame
            crop = frame[y1:y2, x1:x2]

            # Extract feature from cropped player image (dummy feature vector)
            feat = self.feature_extractor(crop)

            matched_id = None
            best_sim = 0.7  # Similarity threshold for matching
            # Try to match current detection with existing tracked players
            for pid, (old_box, old_feat) in self.players.items():
                sim = cosine_similarity(feat, old_feat)  # Compute similarity
                if sim > best_sim:
                    best_sim = sim
                    matched_id = pid  # Match found

            # If no matching ID found, assign a new one
            if matched_id is None:
                matched_id = self.next_id
                self.next_id += 1

            # Save the new/updated player info
            new_players[matched_id] = ([x1, y1, x2, y2], feat)

            # Draw the bounding box with ID on the frame
            draw_bbox(frame, x1, y1, x2, y2, matched_id)

        # Update internal state with the new player set
        self.players = new_players

        return frame  # Return the annotated frame
