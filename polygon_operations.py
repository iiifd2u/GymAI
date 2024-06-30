import os
import traceback

import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from shapely import Polygon, MultiPolygon, Point
from scipy.sparse import coo_matrix
from typing import Tuple, List

weights = os.path.join("models", "yolo_models","segmentation", "yolov8m-seg.pt")
model = YOLO(weights)
w, h = (300, 700)

def predict_on_image(model, img, conf=0.5):
    """Предсказывает для первого входящего Person всю информацию"""

    result = model.predict(img, conf=conf, save=False, device = "cpu")[0]
    print(f"Result: {result}")
    try:
        cls = result.boxes.cls.cpu().numpy()    # cls, (N, 1)
        probs = result.boxes.conf.cpu().numpy()  # confidence score, (N, 1)
        boxes = result.boxes.xyxy.cpu().numpy()   # box with xyxy format, (N, 4)
        masks = result.masks.xy

        # masks = result.masks.data

        predicted_classes = [result.names[c] for c in cls]
        for i, c in enumerate(predicted_classes):
            if c=="person":
                mask, prob, box = masks[i], probs[i], boxes[i]
                box = [int(el) for el in box]

                return box, mask, cls, prob, result.orig_shape
    except Exception as e:
        print(e)
        print(traceback.format_exc())


def get_IOU(pol1 :Polygon, pol2 :Polygon):

    pol_i = pol1.intersection(pol2)  # .geoms[0] - может быть массив геометрий
    if isinstance(pol_i, MultiPolygon):
        pol_i = pol_i.geoms[0]
    pol_u = pol1.union(pol2)  # .geoms[0] - может быть массив геометрий
    if isinstance(pol_u, MultiPolygon):
        pol_u = pol_u.geoms[0]

    return pol_i.area /pol_u.area

def draw_polygon(pol :Polygon, h=700, w = 300):

    x, y = pol.exterior.xy
    image = np.zeros(shape=(h, w, 3))
    pts = np.vstack([y, x]).T
    cv2.fillPoly(image, pts=np.int32([pts]), color=(255, 0, 0))
    plt.imshow(image)
    plt.show()

def draw_polygon_on_real_img(real_img :np.ndarray, pol :Polygon, poligon_shape :Tuple):
    """Передать реальное изображение и другой полигон, посмотреть как он ложиться на силуэт"""

    result = predict_on_image(model, real_img)  # рамка, маска(контур объекта), _, вероятность, размер изображения
    if result is not None:
        box, mask, _, prob, orig_shape = result
        alpha = 0.3
        wp, hp = poligon_shape
        hr, wr = real_img.shape[:2]
        x, y = pol.exterior.xy

        # делим размеры коробки на размеры полигональной маски
        scale_y = int(box[3] - box[1] ) / hp
        scale_x = int(box[2] - box[0] ) / wp

        print(f"scale x: {scale_x}, scale y: {scale_y}")

        resized_mask = np.vstack([np.array(x, dtype=np.int32), np.array(y, dtype=np.int32)]).T
        print(f"max x = {resized_mask[:, 0].max()} : {int(box[2] - box[0])} | max y = {resized_mask[:, 1].max()} : {int(box[3] - box[1])} ")

        data = np.ones(resized_mask.shape[0]) * 255 # контур делаем белым цветом

        mask = coo_matrix((data, (resized_mask[:, 0], resized_mask[:, 1])), shape=(int(box[3] - box[1]), int
            (box[2] - box[0]))) # по координатам строим разреженную матрицу, чтобы получить нужный размер обрезка
        mask = cv2.resize(mask.toarray(), dsize=(w, h)) # обрезанная под пешехода область кадра
        print(mask.shape)
        exit()



        image_overlay = np.zeros((hr, wr))

        image_combined = cv2.addWeighted(real_img, 1 - alpha, image_overlay, alpha, 0)

def create_polygon(img :np.ndarray, h = 700, w = 300 )->Polygon:

    result = predict_on_image(model, img) # рамка, маска(контур объекта), _, вероятность, размер изображения
    if result is not None:
        box, mask, _, prob, orig_shape = result
        w_cur, h_cur = int(box[2] - box[0]), int(box[3] - box[1]) # размеры изображения внутри ограничения
        mask_cropped = np.vstack \
            ([mask[:, 1] - box[1], mask[:, 0] - box[0]]).T  # H, W координаты начинают считататься от нижнего угла рамки,
        # а не изображения

        print(f"Размеры маски до: H {mask_cropped[:, 0].max()} | W {mask_cropped[:, 1].max()}")

        scale_y = (h - 2) / h_cur # коэффициент рескейла
        scale_x = (w - 2) / w_cur

        print(f"Коэффициенты масштабирования {scale_x} | {scale_y}")


        # перемастабированная маска в виде координат границы
        mask_cropped = np.vstack([list(map(lambda el: math.floor(el), mask_cropped[:, 0] * scale_y)),
                                  list(map(lambda el: math.floor(el), mask_cropped[:, 1] * scale_x))]).T

        print(f"Размеры маски после: H {mask_cropped[:, 0].max()} | W {mask_cropped[:, 1].max()}")

        polygon = Polygon(mask_cropped.tolist())
        return polygon