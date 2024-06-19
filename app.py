import os
import cv2
import torch
import configparser
from ultralytics import YOLO

root = os.getcwd()

config = configparser.ConfigParser()
config.read(os.path.join("configfiles", "conf.ini"))
# Инициализация переменных
model_weights = config["paths"]["model_weights"]


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f"Используется на устройстве {device}")

# models_weights = os.path.join("models", "yolo_models", "yolov8n-pose.pt")
model = YOLO(model_weights)


def streaming(model, videofilepath:str):
    cap = cv2.VideoCapture(videofilepath)

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            results = model.predict(frame, save=False)
            annotated_frame = results[0].plot()
            cv2.imshow("YOLOv8 Inference", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    source = os.path.join("data", "loaded_videos", "Watch A Flawless Pommel Horse Performance By Alec Yoder.mp4")
    streaming(model, source)

