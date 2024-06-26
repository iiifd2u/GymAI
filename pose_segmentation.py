import os
import cv2
import random
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Загрузка модели YOLOv8
model = YOLO(os.path.join("models", "yolo_models", "segmentation", 'yolov8m-seg.pt'))

colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255)
]


def process_image(image_path, color):

    os.makedirs('result_segments', exist_ok=True)


    image = cv2.imread(image_path)
    image_origig = image.copy()
    h_orig, w_orig = image.shape[:2]
    # image = cv2.resize(image, (480, 640))
    results = model(image)[0]

    classes_names = results.names
    classes = results.boxes.cls.cpu().numpy()
    masks = results.masks.data.cpu().numpy()
    boxes = results.boxes.xyxy.cpu().numpy() #left top right bottom

    images = []


    # Наложение масок на изображение
    for i, mask in enumerate(masks):
        print(classes_names[classes[i]])

        if classes_names[classes[i]] =="person":
            # color = random.choice(colors)
            coords  = boxes[i]
            x_l, y_t, x_r, y_b = [int(el) for el in coords]
            x = abs(x_r-x_l)
            y = abs(y_t-y_b)
            # print(coords)

            mask_resized = cv2.resize(mask, (w_orig, h_orig))

            color_mask = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
            color_mask[mask_resized > 0] = color
            color_mask = color_mask[y_t:y_b, x_l:x_r]

            mask_filename = os.path.join('result_segments', f"{classes_names[classes[i]]}_{i}.png")


            # image_origig = cv2.addWeighted(image_origig, 1.0, color_mask, 0.5, 0)
            inversed_img = cv2.bitwise_not(color_mask)
            return inversed_img
            # cv2.imwrite(mask_filename, inversed_img)
            # cv2.imshow("0", inversed_img)
            # cv2.waitKey(0)



    # new_image_path = os.path.join('results', os.path.splitext(os.path.basename(image_path))[0] + '_segmented' +
    #                               os.path.splitext(image_path)[1])
    # cv2.imwrite(new_image_path, image_origig)
    # print(f"Segmented image saved to {new_image_path}")

data_root = r"data\datasets\male\Floor\Front Flip"

# file = os.path.join(data_root, random.choice(os.listdir(data_root)))
images = []

fig, ax = plt.subplots(1, 6)
for file in os.listdir(data_root):
    img = process_image(os.path.join(data_root, file), [255, 255, 255])
    images.append(img)

for i in range(6):
    ax[i].imshow(images[i])
    ax[i].set_axis_off()
plt.show()