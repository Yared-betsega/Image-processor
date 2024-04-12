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

def convert_video_to_images_and_save(video_path, frame_interval):
    output_folder = "./images/"
    # Open the video file
    video = cv2.VideoCapture(video_path)
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Clear the output folder
    shutil.rmtree(output_folder)
    os.makedirs(output_folder)
    
    # Initialize frame count
    frame_count = 0
    
    # Read the first frame
    success, frame = video.read()
    # Loop through the video frames
    while success:
        # Save the frame as an image if it's a necessary frame
        if frame_count % frame_interval == 0:
            image_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(image_path, frame)
        
        # Read the next frame
        success, frame = video.read()
        
        # Increment frame count
        frame_count += 1
    
    # Release the video object
    video.release()
    
    return output_folder
    
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def encode_frame(frame):
    _, encoded_frame = cv2.imencode('.jpg', frame)
    return base64.b64encode(encoded_frame).decode('utf-8')

# Example usage
# video_path = "./video_storage/Why do we dream  Amy Adkins_480p.mp4"
# output_folder = "./converted_images_storage"
# frame_interval = 100  # Extract every 10th frame
# convert_video_to_images(video_path, frame_interval)
