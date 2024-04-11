import requests
import os
from fastapi import FastAPI, UploadFile, File, Form
from typing import List
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from utils import convert_video_to_images, encode_frame, encode_image

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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
    if file.content_type.startswith('video'):
        video_path = f"./videos/{file.filename}"
        await file.seek(0)

        with open(video_path, "wb") as video_file:
            video_file.write(file.file.read())
        image_files = convert_video_to_images(video_path, 100)

    elif file.content_type.startswith('image'):
        image_files = [f"./images/{file.filename}"]
        await file.seek(0)

        with open(image_files[0], "wb") as image_file:
            image_file.write(file.file.read())
    else:
        return {"error": "Invalid file type. Only images and videos are supported."}

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    content = []
    text_question = {
        "type": "text",
        "text": text + ", The answer must be formatted beautifully in HTML. You don't have to use the html tag. You can start with div. Aslo Don't include any description outside the html. "
    }

    content.append(text_question)
    print(content[0]["text"])
    for image_file in image_files:
        if file.content_type.startswith('video'):
            base64_image = encode_frame(image_file)
        else:
            base64_image = encode_image(image_file)

        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })

        if len(content) > 10:
            break
    

    payload = {
        "model": "gpt-4-turbo",
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
        "max_tokens": 300
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    return response.json()
