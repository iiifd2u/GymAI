import os
import cv2
import pickle
import time
from typing import List, Tuple, Dict, Union, Optional
import numpy as np
import pickle
import matplotlib.pyplot as plt
from polygon_operations import create_polygon, \
    draw_polygon, \
    draw_polygon_on_real_img, \
    create_video_from_real_and_ideal, \
    save_gif_with_imageio, numpy_to_polygons, \
    create_combined_images

from video_operations import get_video_duration

class Motion():

    def __init__(self, name,
                 frames_count,
                 frame_size = (700, 300),
                 prefix = "polygons"):
        self.name = name
        self.frames_count = frames_count
        self.frame_size = frame_size
        self.frames = [] # сырые изображения
        self.polygons = [] # сырые полигоны
        self.savename  = f"{name}_{frames_count}_{time.time()}"
        self.prefix = prefix # Для сохранения

    def self_save(self, **kwargs):
        """Сохраняет в новый или дописывает в старый"""
        picklename = os.path.join(self.prefix, self.savename)
        if os.path.exists(picklename):
            with open(picklename, "rb") as f:
                current  = pickle.load(f)
                kwargs.update(current)
        with open(picklename, "wb") as f:
            pickle.dump(kwargs, f)


    def train(self,
              data:Union[List[str], str],
              cropped_st_end:List[Tuple],
              name:str):

        if isinstance(data, str):
            data = (data, )


        timestamps_all = []
        for idx, file in enumerate(data):
            if not os.path.exists(file):
                print("Not found")
                continue
            st_time, end_time = cropped_st_end[idx]
            cropped_video = self.__crop_video_by_nn(file, st_time=st_time, end_time=end_time)
            print(cropped_video)
            # cropped_video = file
            frames = self.split_video_to_fixed_frames(cropped_video, self.frames_count)
            print(len(frames))
            print(frames.shape)

            # for frame in frames:
            #     plt.imshow(frame)
            #     plt.show()

            # np.save(os.path.join("numpies", f"frames_from_video_{name}.npy"), frames)
            self.self_save(frames = frames,
                           name = self.name,
                           timestamps = timestamps_all)


        return timestamps_all

    @staticmethod
    def split_video_to_fixed_frames(video_path, count_frames)->np.ndarray:

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Closed!")
        # all_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # print(all_frames)
        # fps = cap.get(cv2.CAP_PROP_FPS)
        # print(fps)
        #
        # time_step = (all_frames/fps)/count_frames # В секундах
        # print(time_step)
        # step_points = [time_step*el for el in range(count_frames)] # В секундах
        video_duration = get_video_duration(video_path)
        step_points = np.linspace(0, video_duration, count_frames+1)
        print(step_points)
        print(len(step_points))
        step_idx = 0
        frames = []

        while(cap.isOpened()):
            success, frame = cap.read()

            if success:
                t = cap.get(cv2.CAP_PROP_POS_MSEC)/1000 # В секундах
                if t>=step_points[step_idx]:
                    step_idx+=1
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if step_idx > count_frames:
                        break
            else:
                break

        frames = np.array(frames)
        cap.release()
        cv2.destroyAllWindows()
        return frames


    def __crop_video_by_nn(self, video:str, st_time:int, end_time:int, output_folder = "temp")->str:

        cap = cv2.VideoCapture(video)
        name, ext = os.path.basename(video).split(".")

        timestamps = []

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        print(f"nn cap open: {cap.isOpened()}")
        size = (frame_width, frame_height)
        print(size)
        codec = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        output_video = os.path.join(output_folder, name+"_cropped."+ext)
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
                    timestamps.append(t)
            else:
                break
        result.release()
        cap.release()
        cv2.destroyAllWindows()
        return output_video



    def create_classification_polygons_from_numpy(self, numpy_path:str):

        frames = np.load(numpy_path)
        # polygons =[]
        # for frame in frames:
        #     p = create_polygon(frame)
        #     if p is not None:
        #         polygons.append(p)
        polygons = numpy_to_polygons(frames)
        print(f"Количество полигонов: {len(polygons)}")
        self.polygons = polygons
        return polygons


    def create_classification_polygons_from_folder(self, folder1:str, folder2:str):
        images_go = [cv2.imread(os.path.join(folder1, img)) for img in os.listdir(folder1)]
        images_stand = [cv2.imread(os.path.join(folder2, img)) for img in os.listdir(folder2)]
        polygons = []
        for stand, go in zip(images_stand, images_go):
            polygons.append(create_polygon(stand))
            polygons.append(create_polygon(go))
        self.polygons = polygons
        return polygons


    def save_polygons(self, savepath:str):
        with open(savepath, "wb") as f:
            pickle.dump(self.polygons, f)

    def load_polygons(self, loadpath:str):
        with open(loadpath, "rb") as f:
            self.polygons = pickle.load(f)
        return self.polygons

if __name__ == '__main__':

    name = "front_flip"
    flip = Motion(name, frames_count=20) # Создаём класс для тренировки сальто
    videopath = [r"data\loaded_videos\male\Floor\Front Flip\frontflip_tutorial.mp4"]

    flip.train(data=videopath, name=name, cropped_st_end=[(24, 26)])
    exit()
    # это полигоны из видео
    # image = cv2.imread("data/my_videos/screenshot.jpg")


    ###
    # polygons_ideal= flip.create_classification_polygons_from_numpy(os.path.join("numpies", f"frames_from_video_{name}.npy"))
    # savepath = os.path.join("polygons", "somePolygons")
    # flip.save_polygons(savepath)
    # exit()


    ###
    loadpath = os.path.join("polygons", "somePolygons")
    polygons_ideal = flip.load_polygons(loadpath)
    print("len frames ideal =", len(polygons_ideal))
    # for polygon in polygons_ideal:
    #     draw_polygon(polygon)
    # exit()
    frames_real = Motion.split_video_to_fixed_frames(os.path.join("data", "my_videos", "CUTTED_9814.MOV"), count_frames=20)
    print("len frames real =", len(frames_real))
    # exit()

    # for f, p in zip(frames_real, polygons_ideal):
    #     draw_polygon_on_real_img(f, p)
        # draw_polygon(p)
    print(f"Длины массивов: реальный = {len(frames_real)}, идеальный = {len(polygons_ideal)}")
    # Гифка

    # create_video_from_real_and_ideal(real_images=frames_real,
    #                                ideal_polygons=polygons_ideal,
    #                                savepath="output_videos")
    combined_images = create_combined_images(real_images=frames_real, ideal_polygons=polygons_ideal, img_size=(320, 480))
    save_gif_with_imageio(savepath=os.path.join("output_videos", "gifs", "example_3.gif"), combo_images=combined_images)
    # save_gif_with_imageio(real_images=frames_real,
    #                       ideal_polygons=polygons_ideal,
    #                       savepath=os.path.join("output_videos", "gifs", "example_2.gif"),
    #                       gif_size=(320, 480))










