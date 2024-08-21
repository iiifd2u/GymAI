import json
import math

import cv2
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import  StaticFiles
from fastapi import Request, Response
from fastapi.responses import HTMLResponse

from video_operations import get_video_duration, take_screenshot, Seconds

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

fps = 24
video_addr = r"data\my_videos\wheel_A_I.MOV"
video_duration = get_video_duration(videopath=video_addr)
print(math.floor(video_duration*100))
print("duration =", video_duration)

@app.get("/", response_class=HTMLResponse)
def index(request:Request):
    return templates.TemplateResponse(request=request,
                                      name="index.html",
                                      context={"duration":math.floor((video_duration-0.1)*100)})

@app.get("/getVideoDuration")
def get_video_duration():
    print("GET!")
    return json.dumps({"duration":video_duration})

@app.get("/getContentByTime/{side}/", responses={200:{"content":{"image/png":{}}}})
def get_content_by_time(side, timestamp:float = 0):

    cur_frame = take_screenshot(timestamp=Seconds(timestamp), videopath=video_addr)
    res, im_png = cv2.imencode(".png", cur_frame)
    return Response(im_png.tobytes(), media_type="image/png")