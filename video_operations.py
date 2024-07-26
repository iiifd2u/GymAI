import os
import re
import cv2
import numpy as np
import datetime
import subprocess
import traceback

import shlex


"""
Операции с видео через ffmpeg
"""

re_fps = re.compile(r'\d+ fps')
re_size = re.compile(r'\d{3,4}x\d{3,4}')


def get_video_duration(videopath:str):
    try:
        cmd = "ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {}".format(videopath)
        # cmd_spl = shlex.split(cmd)
        cmd_spl = cmd.split(" ")
        process = subprocess.Popen(cmd_spl, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
        print(cmd_spl)
        print(process.stderr)
        return float(process.stdout.read().decode())
    except:
        print(traceback.format_exc())

def take_screenshot(videopath:str, second:int):
    """Скриншот на выбранной секунде"""
    fps = 30
    command = f'ffmpeg -ss {str(datetime.timedelta(seconds=second/fps))} -i {videopath} -vframes 1 -f image2pipe -pix_fmt rgb24 -vcodec rawvideo -'.split(" ")
    ffmpeg = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = ffmpeg.communicate()
    print(err)
    print("fps =", re.search(re_fps, str(err)).group())
    print("size =", re.search(re_size, str(err)).group())
    img = np.frombuffer(out, dtype=np.int8)
    return img

if __name__ == '__main__':

    videopath  = os.path.join("data", "my_videos", "IMG_9814.MOV")
    # screen = take_screenshot(videopath, 5)
    # print(screen.shape)
    #
    # img = cv2.imread(os.path.join("logo", "logo_old.png"))
    # img_new = cv2.resize(img, (320, 180))
    # cv2.imwrite(os.path.join("logo", "logo.png"), img_new)

    print(get_video_duration(videopath))