from ultralytics import YOLO

class PlayerDetector:
    def __init__(self, weights_path):
        self.model = YOLO(weights_path)

    def detect(self, frame):
        results = self.model.predict(source=frame, conf=0.4, verbose=False)
        boxes = []
        for r in results:
            for box in r.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = box
                boxes.append([int(x1), int(y1), int(x2), int(y2), float(conf), int(cls)])
        return boxes
