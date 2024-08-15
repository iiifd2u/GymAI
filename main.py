from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import  StaticFiles
from fastapi import Request
from fastapi.responses import HTMLResponse

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
#
#
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def index(request:Request):
    return templates.TemplateResponse(request=request,
                                      name="index.html",
                                      context={"duration":13})