from imageai.Detection import VideoObjectDetection
import matplotlib.pyplot as plt
import os

execution_path = os.getcwd()

def forFrame(frame_number, output_array, output_count, x):
    print("FOR FRAME ", frame_number)
    print("Output for each object : ", output_array)
    print("Output count for unique objects : ", output_count)
    print("------------END OF A FRAME --------------")

vid_obj_detect = VideoObjectDetection()
vid_obj_detect.setModelTypeAsYOLOv3()
vid_obj_detect.setModelPath(r"C:\Users\wdomc\OneDrive\studia\semestr5\widzenie_komputerowe\projekt_laby\yolov3.pt")
vid_obj_detect.loadModel()

detected_vid_obj = vid_obj_detect.detectObjectsFromVideo(
    input_file_path=r"cat_videos/ragdoll.mkv",
    output_file_path=r"C:\Users\wdomc\OneDrive\studia\semestr5\widzenie_komputerowe\projekt_laby\devon_rex_output.mp4",
    frames_per_second=30,
    log_progress=True,
    per_frame_function=forFrame,
    return_detected_frame=True
)

print(detected_vid_obj)
