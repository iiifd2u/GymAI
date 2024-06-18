import os
import yt_dlp


# Pommel horse
links = ["https://www.youtube.com/watch?v=xWLOo6pPnqE"]
folder_loaded_videos = os.path.join("data", "loaded_videos")

ydl_opts = {
    'paths': {"home":folder_loaded_videos}
    # 'progress_hooks': [my_hook],
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download(links)