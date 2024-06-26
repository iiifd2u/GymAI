import os
import traceback

import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from shapely import Polygon, MultiPolygon, Point, affinity
from scipy.sparse import coo_matrix
from typing import Tuple, List

weights = os.path.join("models", "yolo_models","segmentation", "yolov8m-seg.pt")
model = YOLO(weights)
w, h = (300, 700)

def predict_on_image(model, img, conf=0.5):
    """Предсказывает для первого входящего Person всю информацию"""

    result = model.predict(img, conf=conf, save=False, device = "cpu")[0]
    # print(f"Result: {result}")
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

    hh, ww = pol.exterior.xy
    hs, ws, he, we = pol.bounds
    image = np.zeros(shape=(h, w, 3), dtype = np.uint8)
    pts = np.vstack([ww-ws*np.ones(len(ww)), hh-hs*np.ones(len(hh))]).T
    cv2.fillPoly(image, pts=np.int32([pts]), color=(255, 0, 0))
    plt.imshow(image)
    plt.show()

def draw_polygon_on_real_img(real_img :np.ndarray, pol :Polygon, h=700, w = 300):
    """Передать реальное изображение и другой полигон, посмотреть как он ложиться на силуэт"""

    result = predict_on_image(model, real_img)  # рамка, маска(контур объекта), _, вероятность, размер изображения
    if result is not None:
        box, mask_real, _, prob, orig_shape = result
        alpha = 0.3

        mask_real = np.vstack([mask_real[:, 0]- box[0], mask_real[:, 1]- box[1]]).T
        pol_real = Polygon(mask_real.tolist())


        hs, ws, he, we = pol.bounds
        hr, wr = real_img.shape[:2]
        y, x = pol.exterior.xy

        # делим размеры коробки на размеры полигональной маски
        scale_h = int(box[3] - box[1] ) / h
        scale_w = int(box[2] - box[0] ) / w

        x_normalised = (x-ws*np.ones(len(x)))*scale_w
        y_normalised = (y - hs * np.ones(len(y)))*scale_h

        x_normalised = x_normalised.astype(np.int32)
        y_normalised = y_normalised.astype(np.int32)

        mask = np.vstack([x_normalised, y_normalised]).T
        mask_img = np.zeros((int(box[3]-box[1]), int(box[2]-box[0]), 3))
        mask_img_real = np.zeros((int(box[3]-box[1]), int(box[2]-box[0]), 3))

        cv2.fillPoly(mask_img, pts=np.int32([mask]), color=(255, 0, 0))
        cv2.fillPoly(mask_img_real, pts=np.int32([mask_real]), color=(0, 255, 0))

        left = box[0]
        bottom = hr - box[3]
        right = wr - box[2]
        top = box[1]

        masked_img = cv2.copyMakeBorder(mask_img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                        value=(0, 0, 0)).astype(np.uint8)
        masked_img_real = cv2.copyMakeBorder(mask_img_real, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                        value=(0, 0, 0)).astype(np.uint8)

        # image = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB).astype(np.uint8)
        image = real_img.astype(np.uint8)
        image_red_green = cv2.addWeighted(masked_img, 1, masked_img_real, 1, 0)

        image_combined = cv2.addWeighted(image, 1 - alpha, image_red_green, alpha, 0)
        plt.imshow(image_combined)
        plt.show()

def create_polygon(img :np.ndarray, h = 700, w = 300 )->Polygon:

    result = predict_on_image(model, img) # рамка, маска(контур объекта), _, вероятность, размер изображения
    if result is not None:
        box, mask, _, prob, orig_shape = result
        mask_cropped = np.vstack([mask[:, 1] - box[1], mask[:, 0] - box[0]]).T  # H, W координаты начинают считататься от нижнего угла рамки,
        polygon = Polygon(mask_cropped.tolist())
        bounds = polygon.bounds
        h_cur, w_cur = (bounds[2]-bounds[0], bounds[3]-bounds[1])
        print(f"polygon before w x h = {w_cur} x {h_cur}")

        scale_h = (h) / h_cur # коэффициент рескейла
        scale_w = (w) / w_cur

        print(f"Коэффициенты масштабирования w: {scale_w} | h:{scale_h}")

        polygon  =affinity.scale(polygon, xfact=scale_h, yfact=scale_w)
        bounds = polygon.bounds
        h_cur, w_cur = (bounds[2] - bounds[0], bounds[3] - bounds[1])
        print(f"polygon after w x h = {w_cur} x {h_cur}")

        # # перемастабированная маска в виде координат границы
        # mask_cropped = np.vstack([list(map(lambda el: math.floor(el), mask_cropped[:, 0] * scale_y)),
        #                           list(map(lambda el: math.floor(el), mask_cropped[:, 1] * scale_x))]).T
        #
        # print(f"Размеры маски после: H {mask_cropped[:, 0].max()} | W {mask_cropped[:, 1].max()}")
        #
        # polygon = Polygon(mask_cropped.tolist())
        return polygon