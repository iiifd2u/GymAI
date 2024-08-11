import gradio as gr

"""
Заливаем видео, указываем старт-стоп временные метки, элемент, ждём появление видео справа
"""

# Setup for Gradio interface
model_name = gr.Radio([], value="YOLOv3", label="Model", info="choose your model")
input_video = gr.Video(sources=None, label="Input Video")
output_frame = gr.Image(type="numpy", label="Output Frames")
output_video_file = gr.Video(label="Output video")

# Create Gradio Interface for Video Inference
interface_video = gr.Interface(
    fn=func,
    inputs=[input_video, model_name],
    outputs=[output_frame, output_video_file],
    title="Video Inference",
    description="Upload your video and select one model and see the results!",
    examples=[["sample/video_1.mp4"], ["sample/person.mp4"]],
    cache_examples=False,
)