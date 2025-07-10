from utils import draw_bbox, cosine_similarity
import numpy as np
import cv2

class PlayerTracker:
    def __init__(self):
        self.next_id = 0
        self.players = {}  # id: [bbox, feature]
        self.feature_extractor = lambda crop: cv2.resize(crop, (64, 128)).flatten()  # dummy feature

    def update(self, frame, detections, frame_id):
        new_players = {}
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            crop = frame[y1:y2, x1:x2]
            feat = self.feature_extractor(crop)

            matched_id = None
            best_sim = 0.7  # similarity threshold
            for pid, (old_box, old_feat) in self.players.items():
                sim = cosine_similarity(feat, old_feat)
                if sim > best_sim:
                    best_sim = sim
                    matched_id = pid

            if matched_id is None:
                matched_id = self.next_id
                self.next_id += 1

            new_players[matched_id] = ([x1, y1, x2, y2], feat)
            draw_bbox(frame, x1, y1, x2, y2, matched_id)

        self.players = new_players
        return frame
