import math
import os
import re
import cv2
import numpy as np
import datetime
import subprocess
import traceback
import shlex
import matplotlib.pyplot as plt
from PIL import Image



"""
Операции с видео через ffmpeg и opencv
"""

re_fps = re.compile(r'\d+ fps')
re_size = re.compile(r'\d{3,4}x\d{3,4}')

class Seconds:
    def __init__(self, number:float):
        self.seconds =math.floor(number)
        self.minutes = self.seconds // 60
        self.seconds = self.seconds%60
        self.microseconds = int((number-self.seconds)*1000)
    def __str__(self):
        return f"00:{str(self.minutes).rjust(2, '0')}:{str(self.seconds).rjust(2, '0')}.{str(self.microseconds).rjust(4, '0')}"
        
        
def get_video_duration(videopath:str):
    cmd = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {videopath}"
    try:
        # cmd_spl = shlex.split(cmd)
        cmd_spl = cmd.split(" ")
        process = subprocess.Popen(cmd_spl, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        data, errors = process.communicate(timeout=20)
        return float(data.decode())
    except:
        print(traceback.format_exc())


def get_w_h_video(videopath:str):
    """ширина и высота кадра видео"""
    cmd = f"ffprobe -v error -show_entries stream=width,height -of default=noprint_wrappers=1 {videopath}"
    try:
        cmd_spl = cmd.split(" ")
        process = subprocess.Popen(cmd_spl, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        data, errors = process.communicate(timeout=20)
        res = data.decode().split("\n")
        width = int(res[0].split("=")[1])
        height = int(res[1].split("=")[1])
        return (width, height)
    except:
        print(traceback.format_exc())


def take_screenshot(videopath:str, timestamp:Seconds):
    """Скриншот на выбранной секунде"""

    # command = f'ffmpeg -ss {str(datetime.timedelta(seconds=second/fps))} -i {videopath} -vframes 1 -f image2pipe -pix_fmt rgb24 -vcodec rawvideo -'.split(" ")
    cmd = f"ffmpeg -i {videopath} -c:v rawvideo -pix_fmt rgb24 -ss {str(timestamp)} -frames:v 1 -f image2pipe -vcodec rawvideo -"

    try:

        cmd_spl = cmd.split(" ")
        (width, height) = get_w_h_video(videopath)

        process = subprocess.Popen(cmd_spl, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        data, errors= process.communicate(timeout=20)
        np_output = np.fromiter(data, dtype=np.uint8).reshape((height, width, 3))
        return np_output

    except Exception as e:
        print(traceback.format_exc())
    # print(err)
    # print("fps =", re.search(re_fps, str(err)).group())
    # print("size =", re.search(re_size, str(err)).group())
    # img = np.frombuffer(out, dtype=np.int8)
    # return img

def split_video_to_fixed_frames(videopath, frames_count)->np.ndarray:

    cap = cv2.VideoCapture(videopath)
    if not cap.isOpened():
        print("Closed!")
    video_duration = get_video_duration(videopath)
    step_points = np.linspace(0, video_duration, frames_count+1)
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
                if step_idx > frames_count:
                    break
        else:
            break

    frames = np.array(frames)
    cap.release()
    cv2.destroyAllWindows()
    return frames

def loop_gif(gifpath:str):
    with Image.open(gifpath) as gif:
        # print(gif.info)
        gif.info['loop'] = 10
        gif.save(os.path.join("output_videos", "gifs", "looping.gif"), 'GIF')

if __name__ == '__main__':

    videopath  = os.path.join("data", "my_videos", "IMG_9814.MOV")
    # videopath = r"data\my_videos\IMG_9814.MOV"
    # screen = take_screenshot(videopath, 5)
    # print(screen.shape)
    #
    # img = cv2.imread(os.path.join("logo", "logo_old.png"))
    # img_new = cv2.resize(img, (320, 180))
    # cv2.imwrite(os.path.join("logo", "logo.png"), img_new)

    # print(get_video_duration(videopath))
    # print(get_w_h_video(videopath))
    # img = take_screenshot(videopath=videopath, timestamp=Seconds(7.0))
    # plt.imshow(img)
    # plt.show()

    loop_gif(os.path.join("output_videos", "gifs", "cartwheel.gif"))