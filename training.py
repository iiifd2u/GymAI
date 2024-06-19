import os
import random
import re
import cv2
import time
import json
import torch
import pickle
import imageio
import numpy as np
import configparser
from PIL import Image
from sklearn import svm
from ultralytics import YOLO
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.sparse import coo_matrix
from typing import List, Tuple, Dict, Union
from scipy.spatial import ConvexHull
from sklearn.neighbors import NearestNeighbors
from shapely import Polygon, MultiPolygon, Point
import pandas as pd
from sklearn import tree
from catboost import Pool, CatBoostClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score

root = os.getcwd()

config = configparser.ConfigParser()
config.read(os.path.join("configfiles", "conf.ini"))

folder_model = config["paths"]["folder_model"]
folder_output_matrices = config["paths"]["folder_output_matrices"]
folder_input_imgs = config["paths"]["folder_input_imgs"]
name_of_summary_matrix = config["paths"]["summary_matrix"]
folder_classificators = config["paths"]["folder_classificators"]
# Гиперпараметры
grid_step = float(config["hyperparams"]["grid_step"]) # шаг сетки заполнения пустого пространства
points_conf_level = float(config["hyperparams"]["points_conf_level"]) # вклад попадания точек в свои области, тогда вклад совпадения углов = 1 - points_conf_level
std_for_angle = float(config["hyperparams"]["std_for_angle"]) # Градусное отклонение от модели
kernel_SVC = config["hyperparams"]["kernel_SVC"] #Ядро
gamma_SVC = float(config["hyperparams"]["gamma_SVC"])
C_SVC = float(config["hyperparams"]["C_SVC"])

points_dict = dict()
for tag in ["left leg", "right leg", "left hand", "right hand"]:
    points_dict[tag] = json.loads(config["points"][tag])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Используется на устройcтве", device)
yolo_model = YOLO(folder_model)

font = cv2.FONT_HERSHEY_COMPLEX
fontScale = 1
thickness = 2
bottomLeftCornerOfText = (10,20)
fontColor = (0,0,255)
lineType = 2

def get_angles_from_coord_dicts(points_to_angle:Dict, limbs:List):
    """На вход получает словарь с ключами в виде номере точки и значениями в виде координат (х, у)"""
    angles ={}

    if "left leg" in limbs:
    # Углы по теореме косинусов:
        AC = np.sqrt(
            (points_to_angle[12][0] - points_to_angle[16][0]) ** 2 + (points_to_angle[12][1] - points_to_angle[16][1]) ** 2)
        AB = np.sqrt(
            (points_to_angle[12][0] - points_to_angle[14][0]) ** 2 + (points_to_angle[12][1] - points_to_angle[14][1]) ** 2)
        BC = np.sqrt(
            (points_to_angle[14][0] - points_to_angle[16][0]) ** 2 + (points_to_angle[14][1] - points_to_angle[16][1]) ** 2)
        try:
            angles['left leg'] = np.rad2deg(np.arccos((AB ** 2 + BC ** 2 - AC ** 2) / (2 * AB * BC)))
        except ZeroDivisionError as zde:
            print("Ошибка: ", zde)

    if "right leg" in limbs:
        AC = np.sqrt(
            (points_to_angle[11][0] - points_to_angle[15][0]) ** 2 + (points_to_angle[11][1] - points_to_angle[15][1]) ** 2)
        AB = np.sqrt(
            (points_to_angle[11][0] - points_to_angle[13][0]) ** 2 + (points_to_angle[11][1] - points_to_angle[13][1]) ** 2)
        BC = np.sqrt(
            (points_to_angle[13][0] - points_to_angle[15][0]) ** 2 + (points_to_angle[13][1] - points_to_angle[15][1]) ** 2)
        try:
            angles['right leg'] = np.rad2deg(np.arccos((AB ** 2 + BC ** 2 - AC ** 2) / (2 * AB * BC)))
        except ZeroDivisionError as zde:
            print("Ошибка: ", zde)

    return angles

def get_predict(path:Union[str, np.ndarray], points_to_collect:List[int])->Tuple[np.ndarray, List, List, np.ndarray, Dict]:
    """Возвращает кортеж (оригинальное изображение, точки в реальном масштабе, координаты для универсальной рамки(400х200), лейблы точек)"""

    result = yolo_model(path)  # Предсказание


    points_to_angle = {} # cловарь для углов
    points_to_draw = []  # точки для отрисовки
    labels = []
    # angles = {} # словарь углов

    # сначала создаём пустое изображение(матрицу), которую заполним нужными точками
    w, h, _ = result[0].orig_img.shape
    zero_box = result[0].boxes.xywh[0]
    start_pt = (int(zero_box[0] - zero_box[2] / 2), int(zero_box[1] - zero_box[3] / 2))  #
    end_pt = (int(zero_box[0] + zero_box[2] / 2), int(zero_box[1] + zero_box[3] / 2))  #

    """Замена логики распознавания точек для гарантированной связанности лэйблов и координат"""
    for idx_pt, xy in enumerate(result[0].keypoints.xy[0]):  # итерация по точкам для 0-ой рамки
        if idx_pt in points_to_collect:  # точки, отвечающие за ноги
            if xy != [0., 0.]:
                points_to_draw.append([int(xy[0]), int(xy[1])])  # X и Y положения
                points_to_angle[idx_pt]=[int(xy[0]), int(xy[1])] #ключ - номер точки, значение - пары координат
                labels.append(idx_pt)

    labels = np.array(labels)

    w, h = end_pt[0] - start_pt[0], end_pt[1] - start_pt[1]  # размеры ограничивающей рамки , ХУ
    ratio_w, ratio_h = 200 / w, 400 / h  # коэффициенты ширины и высоты

    # точки для предикта, отмасштабированные, el - пара точек х и у
    points_to_predict = list(map(lambda el: [int(ratio_w * (el[0] - start_pt[0])),
                                             int(ratio_h * (el[1] - start_pt[1]))], points_to_draw))

    limbs = []
    for k, v in points_dict.items():
        if set(points_to_collect).issuperset(v):
            limbs.append(k)
    angles = get_angles_from_coord_dicts(points_to_angle, limbs)

    return (result[0].orig_img, points_to_draw, points_to_predict, labels, angles)

def convert_classes_to_real_labels(classes:List[int], real_labels:List[int], cluster_centers:np.ndarray)->Dict:
    """
    Переводит классы в реальные лейблы, возвращает схему перевода
    """

    schema ={} #схема перевода классов в лейблы
    temp =[el.tolist() for el in cluster_centers]
    for i in range(len(classes)):
        temp[i].append(classes[i])

    # сортируем по высоте:
    sorted_positions = list(sorted(temp, key=lambda x:x[1]))
    # Две первых - однозначно точки 11 и 12
    if sorted_positions[0][0] < sorted_positions[1][0]:
        schema[sorted_positions[0][2]] = real_labels[1]
        schema[sorted_positions[1][2]] = real_labels[0]
    else:
        schema[sorted_positions[0][2]] = real_labels[0]
        schema[sorted_positions[1][2]] = real_labels[1]

    # TODO: Далее пока что работаем по предположению, что левая нога однозначно слева, правая - справа,
    # TODO: кроме того, колени всегда ниже бёдер и выше коньков

    # 3, 4 по счёту - однозначно точки 13 и 14
    if sorted_positions[2][0] < sorted_positions[3][0]:
        schema[sorted_positions[2][2]] = real_labels[3]
        schema[sorted_positions[3][2]] = real_labels[2]
    else:
        schema[sorted_positions[2][2]] = real_labels[2]
        schema[sorted_positions[3][2]] = real_labels[3]

    # 5, 6 по счёту - однозначно точки 15 и 16
    if sorted_positions[4][0] < sorted_positions[5][0]:
        schema[sorted_positions[4][2]] = real_labels[5]
        schema[sorted_positions[5][2]] = real_labels[4]
    else:
        schema[sorted_positions[4][2]] = real_labels[4]
        schema[sorted_positions[5][2]] = real_labels[5]

    print("Сортированные позиции:", sorted_positions)

    return schema

def simulator_many_images(person, pose, phase, name_of_summary_matrix:str, count =50 ):
    """Иммитирует выборку из разных изображений по одному изображению"""

    os.makedirs(os.path.join(folder_output_matrices, person), exist_ok=True)
    os.makedirs(os.path.join(folder_output_matrices, person, pose), exist_ok=True)
    os.makedirs(os.path.join(folder_output_matrices, person, pose, phase), exist_ok=True)
    path_to_summary_matrix = os.path.join(folder_output_matrices, person, pose, phase, name_of_summary_matrix)

    summary_matrix = np.load(path_to_summary_matrix)
    img = np.where(summary_matrix > 50, summary_matrix, 0)
    sparse = coo_matrix(img)
    print(sparse)
    points = [el for el in zip(sparse.row, sparse.col)]  # YX по изображению
    points = list(sorted(points, key=lambda x: x[0]))

    labels = []
    coord_points = []
    for i in range(count):
        xx = []
        yy = []
        for idx, pt in enumerate(points):
            y, x = pt
            ofs_y = np.random.randint(-15, 15)
            ofs_x = np.random.randint(-20, 20)
            img[y+ofs_y][x+ofs_x] = 255.
            labels.append(idx)
            coord_points.append([x+ofs_x, y+ofs_y])
            xx.append(x+ofs_x)
            yy.append(y+ofs_y)
    coord_points = np.array(coord_points)

    for xy in coord_points:
        img[xy[1]][xy[0]] = 255.
    # plt.imshow(img)
    # plt.show()

    np.save(os.path.join(folder_output_matrices, person, pose, phase, name_of_summary_matrix), img)

def collect_resultimg(person:str, pose:str, phase:str,
                      path_to_images: List[str],
                      points_to_collect:List[int],
                      name_of_summary_matrix:str):

    """Прогнать массив изображений и собрать из них результирующую матрицу по интересующим точкам
    11, 13, 15 - левая нога
    12, 14, 16 - правая нога

    Результирующая матрица сохраняется в выходной папке
    """

    os.makedirs(os.path.join(folder_output_matrices, person), exist_ok=True)
    os.makedirs(os.path.join(folder_output_matrices, person, pose), exist_ok=True)
    os.makedirs(os.path.join(folder_output_matrices, person, pose, phase), exist_ok=True)
    path_to_summary_matrix = os.path.join(folder_output_matrices, person, pose, phase, name_of_summary_matrix)


    #Результирующая матрица содержит все положения опорных точек

    if os.path.exists(path_to_summary_matrix ):
        os.remove(path_to_summary_matrix)
    # else:
    summary_matrix = np.zeros(shape=(400, 200), dtype=np.uint8)

    for path in path_to_images:

        real_img, points_to_draw, points_to_predict, label, angles = get_predict(path, points_to_collect)

        for xy in points_to_predict:
            if xy[1]<400 and xy[0]<200:
                summary_matrix[xy[1]][xy[0]] = 255.
    np.save(path_to_summary_matrix, summary_matrix)

def from_summary_matrix_create_classificator(path_to_summary_matrix:str,
                                             points_to_collect:List[int],
                                             svc_name:str):
    """По итоговой матрице распределения положения нужных точек
    строит модель классификации. Всё, что не попало в область положения
    точек заменяется на равномерную сетку точек, таким образом при
    кластеризации мы получаем надкласс 'не попал в точку'
    """

    summary_matrix = np.load(path_to_summary_matrix)

    sparse = coo_matrix(summary_matrix)
    coord_points = np.array([el for el in zip(sparse.col, sparse.row)])

    count_classes = len(points_to_collect)

    # Заменили DBSCAN на КНН, получили более стабильный результат
    KM = KMeans(n_clusters=len(points_to_collect)).fit(coord_points)  # кластеризация, n класса

    labels_Kmean = KM.labels_
    print("labels_Kmean =", labels_Kmean.shape)
    print("Центры классов", KM.cluster_centers_)
    print("Классы :", set(labels_Kmean))

    #Получаем схему перевода одних точек к другой
    schema = convert_classes_to_real_labels(classes=list(sorted(set(labels_Kmean))),
                                   real_labels=points_to_collect,
                                   cluster_centers=KM.cluster_centers_)
    print('Schema:', schema)

    #Теперь переводим класс из классификатора в настоящую точку:
    angle_points = {schema[point_num]:pair for point_num, pair in enumerate(KM.cluster_centers_)}
    print(angle_points)

    limbs = []
    for k, v in points_dict.items():
        if set(points_to_collect).issuperset(v):
            limbs.append(k)
    # получаем углы для левой и правой ноги
    angles = get_angles_from_coord_dicts(angle_points, limbs=limbs)
    print("Средние углы:", angles)

    # Группы точек
    group_points = [[] for _ in range(count_classes)]  # группы по n классам

    for idx, point in enumerate(coord_points):
        group = KM.predict(np.array(point).reshape((1, -1)))[0]
        group_points[group].append(point)  # Заполняем соответсвующими точками

    group_points = [np.array(el) for el in group_points]

    boundaries = [ConvexHull(group) for group in  group_points] #Выпуклые области по кластеризованным точкам

    polygons = [[] for _ in range(len(boundaries))] #
    for ib, boundary in enumerate(boundaries): #
        for idx in boundary.vertices: #
            polygons[ib].append(group_points[ib][idx]) #
    polygons = [Polygon(poly) for poly in polygons] #список полигонов
    multi_poly = MultiPolygon(polygons)#Мультиполигон, содержащий области, содержащие все "полезные" точки

    # добавляем сетку равномерного пустого пространства
    grid_xx, grid_yy = np.meshgrid(np.arange(0, summary_matrix.shape[1], grid_step),
                                   np.arange(0, summary_matrix.shape[0], grid_step))
    grid = np.c_[grid_xx.ravel(), grid_yy.ravel()]  # координаты точек сетки, равномерно распределённой по изображению
    print("Лейблированные (размер) = ", KM.labels_.shape)
    print("Вся сетка(размер) = ", grid.shape)

    noise_points = np.array(list(filter(lambda point:not multi_poly.contains(Point(point)), grid))) # точки шума, которые не попадают в области полезных точек
    print("Вся сетка после удаления лишних (размер) = ", noise_points.shape)
    all_points_train = np.concatenate([coord_points, noise_points])  # точки сетки, которые определены как фон + целевая выборка
    all_labels_train = np.concatenate([labels_Kmean, -np.ones(shape=noise_points.shape[0])]) # добавляем -1 лейблы для точек шума

    print("Тренировочные лейблы:", set(all_labels_train))

    # all_points_train  - это coord_point + точки фона, поэтому последовательность первых 6ти лейблов совпадает для одних и тех же групп точек
    # это позволяет использовать схему перевода в предикте, поэтому сохраняем её и используем в дальнейшем
    # Для каждой модели своя схема , если предсказан -1 класс, то точку считаем 0

    rbf_svc = svm.SVC(kernel=kernel_SVC, gamma=gamma_SVC, C=C_SVC).fit(all_points_train, all_labels_train)

    """
    чем меньше С, тем больший запас получает разделяющая гиперпоскость, т.е. допускает больше ошибок
    для больших С выбирает гиперплоскость с меньшим запасом и точнее классифицирует
    Выбор записит от степени линейной разделимости данных
    гамма - определяет, на сколько область будет ограничивать конкретные данные, чем меньше - тем больше обобщения данных
    """

    with open(os.path.join("models", "classificators", svc_name), "wb") as f:
        pickle.dump({"model":rbf_svc, "angles":angles, "schema":schema}, f)

    print("Закончен рассчёт", svc_name)

    # Отрисовка для проверки

    h = 1.  # Служебный параметр для отрисовки
    shape = summary_matrix.shape
    x_min, x_max = 0, shape[1]
    y_min, y_max = 0, shape[0]

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = rbf_svc.predict(np.c_[xx.ravel(), yy.ravel()])

    print("Z shape = ", Z.shape)
    plt.scatter(all_points_train[:, 0], all_points_train[:, 1], c=all_labels_train, cmap=plt.cm.coolwarm, s=1)

    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.5)
    plt.axis("on")
    plt.gca().invert_yaxis()
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.max(), yy.min())
    plt.show()

def create_decision_tree(folder_person_pose:str,
                          points_to_collect: List[int], max_images:int):
    dataset = []

    for idx, subdir in enumerate(os.listdir(folder_person_pose)):
        for path in os.listdir(os.path.join(folder_person_pose, subdir))[:max_images]:
            real_img, points_to_draw, points_to_predict, labels, _\
                = get_predict(os.path.join(folder_person_pose, subdir, path), points_to_collect)
            row = [subdir]
            for c in np.array(points_to_predict).ravel():
                row.append(c)
            # print(row)
            dataset.append(row)

    data = pd.DataFrame(data=dataset, columns=["pos", "11x", "11y","12x", "12y","13x", "13y","14x", "14y", "15x", "15y","16x", "16y"])
    # print(data)
    Y = data[["pos"]]
    X = data[["11x", "11y","12x", "12y","13x", "13y","14x", "14y", "15x", "15y","16x", "16y"]]
    print(data)

    clf = tree.DecisionTreeClassifier(max_depth=5)
    clf.fit(X, Y)


    for row in dataset:
        print("true: {} - pred: {}".format(row[0], clf.predict(np.array(row[1:]).reshape(1, -1))))

    with open(os.path.join("models", "trees", "tree_{}".format(max_images)), "wb") as f:
        pickle.dump({"model":clf}, f)
        # for xy in points_to_predict:
        #     if xy[1] < 400 and xy[0] < 200:
        #         summary_matrix[xy[1]][xy[0]] = 255.


def predict_by_tree(tree_path:str):

    with open(tree_path, "rb") as f:
        clf = pickle.load(f)["model"]
    print(clf.get_depth())
    tree.plot_tree(clf)
    plt.show()

def create_ensemble_trees(person:str, pose:str,
                          points_to_collect: List[int], max_images:int):
    """С использование catboost"""
    folder_person_pose = os.path.join(folder_input_imgs, person, pose)
    dataset = []
    val_split_point = int(max_images*3*0.9)
    for idx, subdir in enumerate(os.listdir(folder_person_pose)):
        for path in os.listdir(os.path.join(folder_person_pose, subdir))[:max_images]:
            real_img, points_to_draw, points_to_predict, labels, angles \
                = get_predict(os.path.join(folder_person_pose, subdir, path), points_to_collect)
            row = [subdir]
            for c in np.array(points_to_predict).ravel():
                row.append(c)
            # print(row)
            dataset.append(row)

    data = pd.DataFrame(data=dataset,
                        columns=["pos", "11x", "11y", "12x", "12y", "13x", "13y", "14x", "14y", "15x", "15y", "16x",
                                 "16y"])

    data = data.sample(frac=1).reset_index(drop=True) # перемешать

    data_train, data_val = data[:val_split_point], data[val_split_point:]


    print(data_train)
    print(data_val)
    # exit()
    Y_train = data_train[["pos"]]
    X_train = data_train.loc[:, data_train.columns!="pos"]

    Y_val = data_val[["pos"]]
    X_val = data_val.loc[:, data_val.columns != "pos"]

    dataset_CB_train = Pool(data=X_train, label=Y_train)
    # print(dataset_CB)

    model = CatBoostClassifier(iterations=10,
                               learning_rate=1,
                               depth=2,
                               loss_function='MultiClass')
    # Fit model
    model.fit(dataset_CB_train)
    model.save_model(os.path.join("models", "catboost_models", f"CB_model_{person}_{pose}_{max_poses}"))
    # # Get predicted classes
    preds_class = model.predict(X_val)
    preds_class = np.squeeze(preds_class)
    # print(preds_class.shape)
    true_class = np.squeeze(Y_val.to_numpy())
    acc = accuracy_score(true_class, preds_class)
    recall = recall_score(true_class, preds_class, average=None)

    print(acc)
    print(recall)







def draw(matrix:np.ndarray, centers, shema):

    matrix = np.where(matrix>0, matrix, 1)

    cv_img = cv2.cvtColor(matrix, cv2.COLOR_GRAY2BGR)
    for idx, ptn in enumerate(centers):
        cv_img = cv2.putText(img=cv_img,
                    text=f"{shema[idx]}",
                    org=ptn,
                    fontFace=font,
                    fontScale=0.5,
                    color=fontColor,
                    thickness=thickness,
                    lineType=lineType)
    cv2.imshow("points", cv_img)
    cv2.waitKey(0)



if __name__ == '__main__':

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("Используется на устройcтве", device)

    ############################## Настройка параметров обучения #########################
    points_to_collect = points_dict["left leg"] + points_dict["right leg"]
    points_to_collect.sort()

    person = "first_person"
    pose = "pose_side"
    phase = "otn"

    image_folder = os.path.join(folder_input_imgs, person, pose, phase)
    images_count = 300
    images = [os.path.join(image_folder, file) for file in os.listdir(image_folder)][:images_count]
    ######################################################################################


    # # # Собрать результирующую матрицу
    # collect_resultimg(person=person, pose=pose, phase=phase, path_to_images=images,
    #                   points_to_collect=points_to_collect,
    #                   name_of_summary_matrix=name_of_summary_matrix,
    #                   )
    #
    # # # Делаем классификатор на основе тренировочных данных
    # from_summary_matrix_create_classificator(
    #     path_to_summary_matrix=os.path.join(folder_output_matrices, person, pose, phase, name_of_summary_matrix),
    #     points_to_collect=points_to_collect,
    #     svc_name='{}_{}_{}_model_svc'.format(len(points_to_collect), pose[5:], phase),
    #     )

    # Строим решающее дерево:
    path = os.path.join(folder_input_imgs, person, pose)
    max_poses = 200
    # create_decision_tree(path, points_to_collect, max_poses)
    #
    # tree_path = os.path.join("models", "trees", "tree_{}".format(max_poses))
    # predict_by_tree(path, tree_path)

    # Строим ансамбль
    create_ensemble_trees(person, pose, points_to_collect, max_poses)



