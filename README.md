# 🧠 Player Re-Identification in a Single Feed using YOLOv11

This project solves the task of **Player Re-Identification** from a 15-second sports video using a fine-tuned `YOLOv11` model and a lightweight Re-ID tracking strategy. It ensures that players who exit and re-enter the video frame are **assigned the same ID**.

---

## 🎯 Objective (From Assignment)

> Given a 15-second video (`15sec_input_720p.mp4`), detect all players and assign player IDs.  
> Ensure the same ID is maintained for players who leave and re-enter the frame.  
> Simulate real-time detection and re-identification.

---

## 💡 What I Did

### ✅ 1. **Used provided YOLOv11 model** to detect players:
- Loaded the `.pt` weights via `Ultralytics` API.
- Extracted only **player detections** (ignoring ball or other objects).

### ✅ 2. **Initial IDs assigned** in the first few frames:
- Based on bounding boxes output by YOLO.

### ✅ 3. **Maintained consistent IDs** using visual Re-ID:
- Added a Re-ID method using **color histograms** + **cosine similarity**.
- Compared current frame detections with memory of earlier player appearances.
- If similarity score is high, the same ID is re-used.

### ✅ 4. **Drew bounding boxes + ID labels** and wrote output video.

---

## 🗂️ Folder Structure

```

.
├── input/
│   └── 15sec\_input\_720p.mp4
├── yolov11/
│   └── yolov11\_custom.pt
├── output/
│   └── reid\_result.mp4 (after running)
├── main.py
├── detector.py
├── tracker.py
└── README.md

````

---

## 🛠️ Setup Instructions

### 1. Clone or download the project

```bash
cd your-folder
````

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate    # for Windows
# OR
source venv/bin/activate  # for Linux/macOS
```

### 3. Install required libraries


    pip install -r requirements.txt


If `requirements.txt` is not available, manually install:

    pip install opencv-python numpy ultralytics


---

## ▶️ Run the Code


    python main.py

It will:

* Load the input video from `input/`
* Detect and track players frame by frame
* Write the output video with IDs to `output/reid_result.mp4`

---

## ✅ Notes

* The custom model is fine-tuned for **player and ball detection** using YOLOv11.
* Tracking logic is based on color appearance, NOT deep embeddings — hence fast and lightweight.
* Works on re-appearing players even after occlusion or temporary disappearance.

---

## 📈 Output Example

![Example](output/sample_frame.jpg) *(Optional)*
Each player is labeled with a unique ID that **remains consistent throughout**.

---

## 🧠 Future Work

* Add trajectory tracking or path visualization.
* Integrate DeepSORT or ReIDNet for high-accuracy identity tracking.
* Expand to multiple camera feeds or real-time streaming.

---

## 🙌 Author

**Kishan Kumar**
BTech Student, NIT Hamirpur
Project done as part of assignment: *“Re-Identification in a Single Feed”*

---

