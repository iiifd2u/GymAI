import os
import cv2
import uuid

class Dataset():
    """Создаёт датасет по видеофайлу"""
    def __init__(self, videopath):
        self.videopath = videopath
        self.__get_all_folders()

    def __get_all_folders(self):
        path = os.path.normpath(self.videopath)
        splitted = path.split(os.sep)
        self.gender, self.apparatus, self.type_ex, self.videoname = splitted[8:12]

    def create(self):
        self.savepath = os.path.join("data", "datasets", self.gender, self.apparatus, self.type_ex)
        unique_name = str(uuid.uuid4())
        os.makedirs(self.savepath, exist_ok=True)

        cap = cv2.VideoCapture(self.videopath)
        frame_num = 0
        while cap.isOpened():
            success, frame = cap.read()

            if success:
                cv2.imwrite(os.path.join(self.savepath, unique_name + f"_{frame_num}.jpg"), frame)
                frame_num+=1
            else:
                break

        cap.release()
        cv2.destroyAllWindows()
        print("created!")

    def __str__(self):
        return self.videopath



if __name__ == '__main__':
    path = r"C:\Users\iii\Documents\Programm\GymAI\data\loaded_videos\male\Floor\Front Flip\Standing Frontflip Tutorial [iENtDEeocLQ].webm"
    ds_FF = Dataset(path)
    ds_FF.create()