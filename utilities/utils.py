import base64
import cv2
import os
import shutil


def convert_video_to_images(video_path, frame_interval):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        base64Frames.append(encode_frame(frame))

    video.release()

    return base64Frames


def convert_video_to_images_for_aws(video_path, frame_interval):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, image_buffer = cv2.imencode('.jpg', frame)
        image_bytes = image_buffer.tobytes()
        base64Frames.append(image_bytes)

    video.release()

    return base64Frames




def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def encode_image_for_aws(image_path):
    with open(image_path, "rb") as image_file:
        return image_file.read()


def encode_frame(frame):
    _, encoded_frame = cv2.imencode('.jpg', frame)
    return base64.b64encode(encoded_frame).decode('utf-8')

# Example usage
# video_path = "./video_storage/Why do we dream  Amy Adkins_480p.mp4"
# output_folder = "./converted_images_storage"
# frame_interval = 100  # Extract every 10th frame
# convert_video_to_images(video_path, frame_interval)
