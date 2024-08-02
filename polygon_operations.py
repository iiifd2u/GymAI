import os
import traceback
import cv2
import math
import imageio
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from shapely import Polygon, MultiPolygon, Point, affinity
from scipy.sparse import coo_matrix
from typing import Tuple, List, Optional


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
    print("poli = ", pol_i)
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

def draw_many_polygons(polygons:List[Polygon], h=700, w=300):

    n = len(polygons)
    fig, ax = plt.subplots(n)
    for i in range(n):
        hh, ww = polygons[i].exterior.xy
        hs, ws, he, we = polygons[i].bounds
        image = np.zeros(shape=(h, w, 3), dtype=np.uint8)
        pts = np.vstack([ww - ws * np.ones(len(ww)), hh - hs * np.ones(len(hh))]).T
        cv2.fillPoly(image, pts=np.int32([pts]), color=(255, 0, 0))
        ax[i].imshow(image)
    ax[0].set_xlabel("ideal")
    ax[1].set_xlabel("real")
    plt.show()

def create_combined_image(real_img :np.ndarray, pol_ideal :Polygon, h=700, w = 300)->np.ndarray:
    """Передать реальное изображение и другой полигон, посмотреть как он ложиться на силуэт"""

    result = predict_on_image(model, real_img)  # рамка, маска(контур объекта), _, вероятность, размер изображения
    if result is not None:
        box, mask_real, _, prob, orig_shape = result
        alpha = 0.3

        mask_real = np.vstack([mask_real[:, 0]- box[0], mask_real[:, 1]- box[1]]).T
        # pol_real = Polygon(mask_real.tolist())

        hs, ws, he, we = pol_ideal.bounds
        hr, wr = real_img.shape[:2]
        y, x = pol_ideal.exterior.xy

        # делим размеры коробки на размеры полигональной маски
        scale_h = int(box[3] - box[1]) / h
        scale_w = int(box[2] - box[0]) / w

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
        # plt.imshow(masked_img)
        # plt.show()
        # plt.imshow(masked_img_real) # вот это надо добавить
        # plt.show()

        # image = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB).astype(np.uint8)
        image = real_img.astype(np.uint8)
        image_red_green = cv2.addWeighted(masked_img, 1, masked_img_real, 1, 0)
        image_combined = cv2.addWeighted(image, 1 - alpha, image_red_green, alpha, 0)

        ################### КУСОК ДЛЯ ПОСТРОЕНИЯ ПОЛИГОНА ###########################
        mask_cropped = np.vstack(
            [mask_real[:, 1] - box[1], mask_real[:, 0] - box[0]]).T  # H, W координаты начинают считататься от нижнего угла рамки,

        polygon = Polygon(mask_cropped.tolist())
        bounds = polygon.bounds
        h_cur, w_cur = (bounds[2] - bounds[0], bounds[3] - bounds[1])
        # print(f"polygon before w x h = {w_cur} x {h_cur}")

        scale_h = (h) / h_cur  # коэффициент рескейла
        scale_w = (w) / w_cur

        # print(f"Коэффициенты масштабирования w: {scale_w} | h:{scale_h}")

        pol_real= affinity.scale(polygon, xfact=scale_h, yfact=scale_w)
        print("pol area =", pol_ideal.area)
        print("pol ideal =", pol_real.area)


        iou = get_IOU(pol_ideal, pol_real)
        draw_many_polygons([pol_ideal, pol_real])
        print(f"iou = {iou}")
        return image_combined

def draw_polygon_on_real_img(real_img :np.ndarray, pol :Polygon, h=700, w = 300):
    image_combined = create_combined_image(real_img, pol, h, w)
    plt.imshow(image_combined)
    plt.show()

def create_combined_images(real_images:List, ideal_polygons:List, img_size:Tuple[int, int]):
    """Из массива изображений и полигонов
    идеального выполнения формирует наложенные друг на друга изображения
    """
    if len(real_images)==len(ideal_polygons):
        combo_images = []

        for real, ideal in zip(real_images, ideal_polygons):
            combo_img = create_combined_image(real_img=real, pol_ideal=ideal, h=h, w=w)
            if combo_img is not None:
                combo_images.append(cv2.resize(combo_img, img_size))
        return combo_images

def save_gif_with_imageio(savepath:str, combo_images:str, duration = 10, fps = 2):
    """Сохраняем гифку"""

    kwargs = {'duration':duration, 'fps':fps}
    imageio.mimsave(uri=savepath, ims=combo_images, format='GIF', **kwargs)

def create_video_from_real_and_ideal(real_images:List, ideal_polygons:List, savepath:str):
    """
    собираем гифку из изображений с наложенными полигонами
    :param real_images:
    :param ideal_images:
    :param savepath:
    :return:
    """

    if len(real_images)==len(ideal_polygons):
        if not os.path.exists(savepath):
            try:
                os.makedirs(savepath, exist_ok=True)
            except Exception as e:
                print("Ошибка из гифки: ", e)
                return None

        combo_images = []
        (width, height) = real_images[0].shape[:2]
        print(f"ширина х высота  = {width} x {height}")
        for real, ideal in zip(real_images, ideal_polygons):
            combo_img = create_combined_image(real_img=real, pol=ideal, h=h, w=w)
            if combo_img is not None:
                combo_images.append(combo_img.astype(np.uint8))

        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fourcc =cv2.VideoWriter_fourcc(*'DIVX')
        video = cv2.VideoWriter(os.path.join(savepath, f"video.avi"), fourcc, 1., (width, height))

        for ci in combo_images:
            video.write(ci)

        video.release()
        cv2.destroyAllWindows()

    else:
        print("Ошибка, массивы различной длины!")

def create_polygon(img :np.ndarray, h = 700, w = 300 )->Optional[Polygon]:

    result = predict_on_image(model, img) # рамка, маска(контур объекта), _, вероятность, размер изображения
    if result is not None:
        box, mask, _, prob, orig_shape = result
        mask_cropped = np.vstack([mask[:, 1] - box[1], mask[:, 0] - box[0]]).T  # H, W координаты начинают считататься от нижнего угла рамки,
        polygon = Polygon(mask_cropped.tolist())
        bounds = polygon.bounds
        h_cur, w_cur = (bounds[2]-bounds[0], bounds[3]-bounds[1])
        # print(f"polygon before w x h = {w_cur} x {h_cur}")

        scale_h = (h) / h_cur # коэффициент рескейла
        scale_w = (w) / w_cur

        # print(f"Коэффициенты масштабирования w: {scale_w} | h:{scale_h}")

        polygon  =affinity.scale(polygon, xfact=scale_h, yfact=scale_w)
        bounds = polygon.bounds
        h_cur, w_cur = (bounds[2] - bounds[0], bounds[3] - bounds[1])
        # print(f"polygon after w x h = {w_cur} x {h_cur}")

        # # перемастабированная маска в виде координат границы
        # mask_cropped = np.vstack([list(map(lambda el: math.floor(el), mask_cropped[:, 0] * scale_y)),
        #                           list(map(lambda el: math.floor(el), mask_cropped[:, 1] * scale_x))]).T
        #
        # print(f"Размеры маски после: H {mask_cropped[:, 0].max()} | W {mask_cropped[:, 1].max()}")
        #
        # polygon = Polygon(mask_cropped.tolist())
        return polygon
    return None

def average_polygon(pol_1:Polygon, pol_2:Polygon)->Polygon:
    """Заглушка"""
    # TODO: Cделать нормальный средний полигон
    return pol_1.intersection(pol_2)

def numpy_to_polygons(frames:np.ndarray)->List[Polygon]:
    """Переводит все кадры в полигон, достраивая до необходимого количества
    искуственные полигоны"""
    polygons = [create_polygon(f) for f in frames]
    #Первым становиться первый нормальный
    for p in polygons:
        if p is None:
            continue
        else:
            polygons[0] = p
            break

    # Последним становится последний нормальный
    for p in polygons[::-1]:
        if p is None:
            continue
        else:
            polygons[-1] = p
            break
    # Костыльное решение
    while None in polygons:
        # print(polygons)
        for idx_p, p in enumerate(polygons[1:-2]):
            idx_p = idx_p+1
            if p is not None:
                continue
            else:

                if polygons[idx_p+1] is not None:
                    polygons[idx_p] = average_polygon(polygons[idx_p-1], polygons[idx_p+1])
                    print(f"AVG p[{idx_p}] = {polygons[idx_p]}")
                else:
                    polygons[idx_p] = polygons[idx_p-1]
                    print(f"OLD p[{idx_p}] = {polygons[idx_p]}")

    return polygons


