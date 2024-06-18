import os
import cv2
import torch
from ultralytics import YOLO

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

print(f"Используется на устройстве {device}")

models_weights = os.path.join("models", "yolo_models", "yolov8n-pose.pt")
model = YOLO(models_weights)

source = os.path.join("data", "loaded_videos", "Watch A Flawless Pommel Horse Performance By Alec Yoder.mp4")

if __name__ == '__main__':

    cap = cv2.VideoCapture(source)

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            results = model.predict(frame, save = False)
            annotated_frame = results[0].plot()
            cv2.imshow("YOLOv8 Inference", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()