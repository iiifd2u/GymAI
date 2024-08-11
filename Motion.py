import os
import cv2
import pickle
import time
import uuid
from datetime import datetime
from typing import List, Tuple, Dict, Union, Optional
import numpy as np
import pickle
import matplotlib.pyplot as plt
from polygon_operations import create_polygon, \
    draw_polygon, \
    draw_polygon_on_real_img, \
    create_video_from_real_and_ideal, \
    save_gif_with_imageio, numpy_to_polygons, \
    create_combined_images,  get_prediction

from video_operations import get_video_duration, split_video_to_fixed_frames

class Motion:

    def __init__(self, name, type:str,
                 prefix = "polygons"):
        """
        Инициализирует тренировочный блок
        :param name: Название элемента
        :param type: Тип: пробный или эталонный ["trial", "master"]
        :param prefix: Папка для сохранения
        """
        self.name = name
        self.uuid = str(uuid.uuid4())
        self.type = type
        self.videotemppath = None #фрагмент, по которому распознаём элемент
        self.frames = [] # сырые изображения
        self.boxes = []
        self.masks = []
        self.polygons = [] # полигоны заданной рамки
        self.savename  = f"{type}_{name}_untrained_dt{datetime.now().strftime('%H_%M_%S')}_{self.uuid}"
        self.prefix = prefix # Для сохранения
        self.timestamps = []



    def self_save(self, **kwargs):
        """Сохраняет в новый или дописывает в старый"""
        picklename = os.path.join(self.prefix, self.savename)
        if os.path.exists(picklename):
            with open(picklename, "rb") as f:
                current  = pickle.load(f)
                kwargs.update(current)
        with open(picklename, "wb") as f:
            pickle.dump(kwargs, f)

    def load(self, picklename:str):
        if os.path.exists(picklename):
            with open(picklename, "rb") as f:
                current  = pickle.load(f)
        self.type = current['type']
        self.name = current["name"]
        self.videotemppath = current["temppath"]
        self.videoduration = current["duration"]
        self.timestamps = current["timestamps"]
        self.frames = current["frames"]
        self.boxes = current["boxes"]
        self.masks = current["masks"]
        self.polygons = current["polygons"]
        self.polygon_size = current["poly_size"]
        self.savename = picklename

    def __cropp_video(self, start:int, end:int):
        try:
            self.videotemppath = self.__crop_video_by_nn(video=self.videopath,
                                                         st_time=start,
                                                         end_time=end)
        except Exception as e:
            print("Ошибка в обрезке видео:", e)

    def __get_frames(self):
        try:
            self.frames = split_video_to_fixed_frames(videopath=self.videotemppath,
                                                      frames_count=self.frames_count)
        except Exception as e:
            print("Ошибка в разбиении видео:", e)

    def __get_polygons(self):
        try:
            self.polygons = numpy_to_polygons(self.frames,
                                              w=self.polygon_size[1],
                                              h= self.polygon_size[0])
            print(f"Количество полигонов: {len(self.polygons)}")
        except Exception as e:
            print("Ошибка при создании полигонов:", e)

    def __get_predictions(self):
        try:
            predictions = [get_prediction(f, h= self.polygon_size[0], w=self.polygon_size[1]) for f in self.frames]
            self.boxes = [el[0] for el in predictions]
            self.masks= [el[1] for el in predictions]
            self.polygons = [el[2] for el in predictions]

            print(f"Количество полигонов: {len(self.polygons)}")
        except Exception as e:
            print("Ошибка при создании полигонов:", e)

    def train(self, videopath:str,  frames_count:int, polygon_size = (700, 300),**kwargs):

        """
        Обрезать видеофрагмент, если необходимо
        Порезать на нужное число кадров
        Прогнать кадры через распознаватор и получить полигоны
        Сохранить этот объект
        """
        self.videopath = videopath
        self.frames_count = frames_count
        self.polygon_size = polygon_size
        start = kwargs.get("start", 0)
        end = kwargs.get("end", 10)

        self.__cropp_video(start, end)
        self.videoduration = get_video_duration(self.videotemppath)
        self.__get_frames()
        # self.__get_polygons()
        self.__get_predictions()
        self.savename = f"{self.type}_{self.name}_fr{frames_count}_dt{datetime.now().strftime('%H_%M_%S')}_{self.uuid}"

        self.self_save(type = self.type,
                       name = self.name,
                       temppath = self.videotemppath,
                       duration = self.videoduration,
                       timestamps = self.timestamps,
                       frames=self.frames,
                       boxes = self.boxes,
                       masks = self.masks,
                       polygons = self.polygons,
                       poly_size = self.polygon_size)


    #
    # def train(self,
    #           data:Union[List[str], str],
    #           cropped_st_end:List[Tuple],
    #           name:str):
    #
    #     if isinstance(data, str):
    #         data = (data, )
    #
    #
    #     timestamps_all = []
    #     for idx, file in enumerate(data):
    #         if not os.path.exists(file):
    #             print("Not found")
    #             continue
    #         st_time, end_time = cropped_st_end[idx]
    #         cropped_video = self.__crop_video_by_nn(file, st_time=st_time, end_time=end_time)
    #         print(cropped_video)
    #         # cropped_video = file
    #         frames = self.split_video_to_fixed_frames(cropped_video, self.frames_count)
    #         print(len(frames))
    #         print(frames.shape)
    #
    #         # for frame in frames:
    #         #     plt.imshow(frame)
    #         #     plt.show()
    #
    #         # np.save(os.path.join("numpies", f"frames_from_video_{name}.npy"), frames)
    #         self.self_save(frames = frames,
    #                        name = self.name,
    #                        timestamps = timestamps_all)
    #
    #
    #     return timestamps_all

    def __crop_video_by_nn(self, video:str, st_time:float, end_time:float, output_folder = "temp")->str:
        """Функция-заглушка, оригинальная функция должна
        обрезать видео по нахождению в ней элемента,
        но пока что это делается по временным рамкам"""
        cap = cv2.VideoCapture(video)
        name, ext = os.path.basename(video).split(".")

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        print(f"nn cap open: {cap.isOpened()}")
        size = (frame_width, frame_height)
        print(size)
        codec = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        output_video = os.path.join(output_folder, name+f"_cropped_dt{datetime.now().strftime('%H_%M_%S')}_{self.uuid}{self.uuid}."+ext)
        result = cv2.VideoWriter(output_video, codec, fps, size)

        start =  st_time*1000
        end = end_time*1000 if end_time>0 else np.inf

        while (cap.isOpened()):
            frame_exists, curr_frame = cap.read()
            if frame_exists:
                t = cap.get(cv2.CAP_PROP_POS_MSEC)
                # Эта строчка заменяет выбор кадра нейросетью, пока что ручной выбор
                if t>start and t<end:
                    result.write(curr_frame)
                    self.timestamps.append(t)
            else:
                break
        result.release()
        cap.release()
        cv2.destroyAllWindows()
        return output_video



    # def create_classification_polygons_from_numpy(self, numpy_path:str):
    #
    #     frames = np.load(numpy_path)
    #     # polygons =[]
    #     # for frame in frames:
    #     #     p = create_polygon(frame)
    #     #     if p is not None:
    #     #         polygons.append(p)
    #     polygons = numpy_to_polygons(frames)
    #     print(f"Количество полигонов: {len(polygons)}")
    #     self.polygons = polygons
    #     return polygons


    # def create_classification_polygons_from_folder(self, folder1:str, folder2:str):
    #     images_go = [cv2.imread(os.path.join(folder1, img)) for img in os.listdir(folder1)]
    #     images_stand = [cv2.imread(os.path.join(folder2, img)) for img in os.listdir(folder2)]
    #     polygons = []
    #     for stand, go in zip(images_stand, images_go):
    #         polygons.append(create_polygon(stand))
    #         polygons.append(create_polygon(go))
    #     self.polygons = polygons
    #     return polygons


    # def save_polygons(self, savepath:str):
    #     with open(savepath, "wb") as f:
    #         pickle.dump(self.polygons, f)
    #
    # def load_polygons(self, loadpath:str):
    #     with open(loadpath, "rb") as f:
    #         self.polygons = pickle.load(f)
    #     return self.polygons
    def __iter__(self):
        """Кортеж из элементов по аналогии с результатом работы нейросети"""
        for row in zip(self.boxes, self.masks, self.frames, self.polygons):
            yield row


def compare_with(master:Motion, trial:Motion, h=700, w=300):
    """Сравнивает два объекта Motion и возвращает кадр trial с наложенным polygon master"""

    for m, t in zip(master, trial):
        try:
            box_master, mask_master, frame_master, polygon_master = m
            box_trial, mask_trial, frame_trial, polygon_trial = t

            alpha = 0.3

            mask_trial_reshaped = np.vstack([mask_trial[:, 0] - box_trial[0], mask_trial[:, 1] - box_trial[1]]).T


            hs, ws, he, we = polygon_master.bounds
            hr, wr = frame_trial.shape[:2]
            y, x = polygon_master.exterior.xy

            # делим размеры коробки на размеры полигональной маски
            scale_h = int(box_trial[3] - box_trial[1]) / h
            scale_w = int(box_trial[2] - box_trial[0]) / w

            x_normalised = (x - ws * np.ones(len(x))) * scale_w
            y_normalised = (y - hs * np.ones(len(y))) * scale_h

            x_normalised = x_normalised.astype(np.int32)
            y_normalised = y_normalised.astype(np.int32)

            mask = np.vstack([x_normalised, y_normalised]).T
            mask_img = np.zeros((int(box_trial[3] - box_trial[1]), int(box_trial[2] - box_trial[0]), 3))
            mask_img_trial = np.zeros((int(box_trial[3] - box_trial[1]), int(box_trial[2] - box_trial[0]), 3))

            cv2.fillPoly(mask_img, pts=np.int32([mask]), color=(255, 0, 0))
            cv2.fillPoly(mask_img_trial, pts=np.int32([mask_trial_reshaped]), color=(0, 255, 0))

            left = box_trial[0]
            bottom = hr - box_trial[3]
            right = wr - box_trial[2]
            top = box_trial[1]

            masked_img = cv2.copyMakeBorder(mask_img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                            value=(0, 0, 0)).astype(np.uint8)
            masked_img_trial= cv2.copyMakeBorder(mask_img_trial, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                                 value=(0, 0, 0)).astype(np.uint8)
            # plt.imshow(masked_img)
            # plt.show()
            # plt.imshow(masked_img_real) # вот это надо добавить
            # plt.show()

            # image = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB).astype(np.uint8)
            image = frame_trial.astype(np.uint8)
            image_red_green = cv2.addWeighted(masked_img, 1, masked_img_trial, 1, 0)
            plt.imshow(image_red_green)
            image_combined = cv2.addWeighted(image, 1 - alpha, image_red_green, alpha, 0)
    #
    # ################### КУСОК ДЛЯ ПОСТРОЕНИЯ ПОЛИГОНА ###########################
    # mask_cropped = np.vstack(
    #     [mask_real[:, 1] - box[1],
    #      mask_real[:, 0] - box[0]]).T  # H, W координаты начинают считататься от нижнего угла рамки,
    #
    # polygon = Polygon(mask_cropped.tolist())
    # bounds = polygon.bounds
    # h_cur, w_cur = (bounds[2] - bounds[0], bounds[3] - bounds[1])
    # # print(f"polygon before w x h = {w_cur} x {h_cur}")
    #
    # scale_h = (h) / h_cur  # коэффициент рескейла
    # scale_w = (w) / w_cur
    #
    # # print(f"Коэффициенты масштабирования w: {scale_w} | h:{scale_h}")
    #
    # pol_real = affinity.scale(polygon, xfact=scale_h, yfact=scale_w)
    # print("pol area =", pol_ideal.area)
    # print("pol ideal =", pol_real.area)
    #
    # iou = get_IOU(pol_ideal, pol_real)
    # draw_many_polygons([pol_ideal, pol_real])
    # print(f"iou = {iou}")
            yield image_combined
        except Exception as e:
            print(f"Внутренняя ошибка {e}")
            yield None


if __name__ == '__main__':

    polygon_size = (700, 300)

    name = "cartwheel"
    masterwheel = Motion(name, type="master")
    videopath = r"data\my_videos\wheel_A_I.MOV"
    masterwheel.train(videopath=videopath,
               frames_count=20,
               polygon_size=polygon_size,
               start=23.5, end=27.7)

    # loadpath = f"polygons\\master_cartwheel_fr20_dt21_10_57"
    loadpath = f"polygons\\{masterwheel.savename}"
    masterwheel.load(loadpath)

    # for p, f in zip(masterwheel.polygons, masterwheel.frames):
    #     f = f/255.
    #     plt.imshow(f)
    #     plt.show()
    #     draw_polygon(p)
    # exit()

    trialwheel= Motion(name, type="trial")  # Создаём класс для тренировки сальто
    # polygon_size = (700, 300)
    videopath = r"data\my_videos\cropped_wheel.MOV"

    trialwheel.train(videopath=videopath,
               frames_count=20,
               polygon_size=polygon_size,
               start=0.7, end=4.5)
    # exit()
    # loadpath = r'polygons\trial_cartwheel_fr20_dt21_33_13'
    loadpath = f'polygons\\{trialwheel.savename}'
    trialwheel.load(loadpath)

    # for p, f in zip(trialwheel.polygons, trialwheel.frames):
    #     f = f/255.
    #     plt.imshow(f)
    #     plt.show()
    #     draw_polygon(p)
    # exit()

    gif_frames = []
    for frame in compare_with(master=masterwheel, trial=trialwheel):
        if frame is not None:
            gif_frames.append(frame)

    save_gif_with_imageio(savepath=os.path.join("output_videos", "gifs", "example_3.gif"), combo_images=gif_frames)
    exit()

    #'ffmpeg -i wheel_A_I_cropped_e79585f1-00d0-4a00-be3d-547181ecf07f.MOV -vf "crop=out_w:out_h:x:y" cropped_wheel.MOV'
    #""

    ########### САЛЬТО ######################


    name = "front_flip"
    masterflip = Motion(name, type="master") # Создаём класс для тренировки сальто


    ################ ЭТАП 1 - обучение эталонных данных
    # videopath = r"data\loaded_videos\male\Floor\Front Flip\frontflip_tutorial.mp4"
    polygon_size = (700, 300)
    #
    # masterflip.train(videopath=videopath,
    #            frames_count=20,
    #            polygon_size=polygon_size,
    #            start=24, end=26)
    # exit()


    ########################## ЭТАП 2 - загрузка готовых данных
    loadpath = r'polygons\master_front_flip_fr20_dt22_59_09'
    masterflip.load(loadpath)
    # print(len(masterflip.masks))
    # print(masterflip.boxes)
    # for p, f in zip(masterflip.polygons, masterflip.frames):
    #     f = f/255.
    #     plt.imshow(f)
    #     plt.show()
    #     draw_polygon(p)
    # exit()

    ################ ЭТАП 3 - обучение тренировочных данных
    videopath = r"data\my_videos\IMG_9814.MOV"
    trialflip = Motion(name, type="trial")  # Создаём класс для тренировки сальто
    # polygon_size = (700, 300)

    # trialflip.train(videopath=videopath,
    #            frames_count=20,
    #            polygon_size=polygon_size,
    #            start=4.5, end=6)
    # exit()

    ################ ЭТАП 4 - загрузка тренировочных данных
    loadpath = r'polygons\trial_front_flip_fr20_dt00_01_34'
    trialflip.load(loadpath)

    # for p, f in zip(trialflip.polygons, trialflip.frames):
    #     f = f/255.
    #     plt.imshow(f)
    #     plt.show()
    #     draw_polygon(p)
    # exit()


    ####################### ЭТАП 5 #########################
    """Сравнение мастера и пробного"""

    gif_frames = []
    for frame in compare_with(master=masterflip, trial=trialflip):
        if frame is not None:
            gif_frames.append(frame)

    save_gif_with_imageio(savepath=os.path.join("output_videos", "gifs", "example.gif"), combo_images=gif_frames)

    exit()












