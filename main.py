import requests
import openai
from openai import OpenAI
import os
from fastapi import FastAPI, UploadFile, File, Form
from typing import List
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from utils import convert_video_to_images, encode_frame, encode_image
import shutil

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process")
async def process_files(text: str = Form(...), file: UploadFile = File(...)):
    is_video = file.content_type.startswith('video')
    
    if is_video:
        path = "./videos"
        if not os.path.exists(path):
            os.mkdir(path)
        shutil.rmtree(path)
        os.mkdir(path)
        video_path = f"{path}/{file.filename}"
        await file.seek(0)

        with open(video_path, "wb") as video_file:
            video_file.write(file.file.read())
        base64Frames = convert_video_to_images(video_path, 100)

    elif file.content_type.startswith('image'):
        path = "./images"
        if not os.path.exists(path):
            os.mkdir(path)
        shutil.rmtree(path)
        os.mkdir(path)
        
        base64Frames = []
        await file.seek(0)
        image_file_loc = f"{path}/{file.filename}"
        with open(image_file_loc, "wb") as image_file:
            image_file.write(file.file.read())
        base64Frames.append(encode_image(image_file_loc))
    else:
        return {"error": "Invalid file type. Only images and videos are supported."}

    gap = len(base64Frames) // 45 if is_video else 1
    start = 25 if is_video else 0
    try: 
        starter = "These are frames from a video that a user want me to process from CCTV. " if is_video else "This is an image a user want me to process. "
        PROMPT_MESSAGES = [
            {
                "role": "user",
                "content": [
                    starter + text + ", The answer must be formatted in HTML. Aslo Don't include any description outside the html. Since I will embed the html in other web app, don't put it inside html or body tags.",
                    *map(lambda x: {"image": x, "resize": 768}, base64Frames[start::gap]),
                ],
            },
        ]
        
        params = {
            "model": "gpt-4-vision-preview",
            "messages": PROMPT_MESSAGES,
            "max_tokens": 200,
        }

        
        result = client.chat.completions.create(**params)

        return result
    
    except openai.BadRequestError as e:
        return e.response.json()
        
