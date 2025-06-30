Brain Tumor Detection and Segmentation:

This project uses the *YOLOv8 object detection model* for tumor localization and the *Segment Anything Model (SAM)* for precise segmentation of brain tumors in MRI images. It supports both single-image and batch inference and is GPU-accelerated for faster performance.

---

Dataset:

* Dataset structure should follow YOLO format.
* The data.yaml file should define class names and image paths:

yaml
train: ../images/train/
val: ../images/val/
test: ../images/test/

nc: 3
names: ['glioma', 'meningioma', 'pituitary']


Ensure your dataset is correctly uploaded to Google Drive or your local environment.

---

###  Features

* Brain tumor detection using *YOLOv8*
*  Segmentation using *Meta's SAM*
*  Supports GPU (CUDA) and CPU
*  Works on individual images or folders
*  Displays output images directly in Google Colab
*  Saves prediction results automatically

---





2. Install dependencies

pip install ultralytics opencv-python matplotlib


3. (Optional for SAM)

pip install segment-anything


###  Training (YOLOv8)

from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # or a custom .yaml file to train from scratch
model.train(
    data="path/to/data.yaml",
    epochs=30,
    imgsz=640,
    device=0  # GPU
)

Inference (YOLO Only)

from ultralytics import YOLO
model = YOLO("runs/detect/train/weights/best.pt")
results = model.predict("t.jpg", conf=0.25, save=True)


#displaying:
To display output in Colab:

from IPython.display import Image, display
display(Image(filename="runs/detect/predict/image.jpg"))



###  Detection + Segmentation (YOLO + SAM)

from ultralytics import YOLO, SAM


# Load models
yolo_model = YOLO("runs/detect/train/weights/best.pt")
sam_model = SAM("sam_b.pt")  # Download from Meta AI if needed


# Run YOLO detection
results = yolo_model("t.jpg")


# Apply SAM on bounding boxes
for result in results:
    boxes = result.boxes.xyxy
    sam_model(result.orig_img, bboxes=boxes, save=True, device="cuda:0")

###  Results

* YOLO detections are saved in runs/detect/predict/
* SAM segmented outputs are saved in runs/segment/predict/

Use the following to show the latest prediction:

import os
from IPython.display import Image, display

output_dir = "runs/detect/predict"
image_file = sorted(os.listdir(output_dir))[-1]
display(Image(filename=os.path.join(output_dir, image_file)))


###  Model Performance

* Track training metrics via results.png in runs/detect/train/
* Loss curves include box loss, cls loss, and dfl loss



###  Troubleshooting

* *No detections?* Check if conf is too high; try conf=0.1
* *Wrong path?* Double-check paths in data.yaml and training call
* *SAM output missing?* Ensure your YOLO detects at least one object



###  Credits

* YOLOv8 by [Ultralytics]
* SAM by [Meta AI]
* Brain MRI Dataset from