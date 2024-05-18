import openai
from openai import OpenAI

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from typing import List
from dotenv import load_dotenv

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import cloudinary
import cloudinary.uploader

import os
import shutil
import datetime

from utilities.utils import convert_video_to_images, convert_video_to_images_for_aws, encode_image_for_aws, encode_image
from utilities.apis import ApiServices
from utilities.response import Response, ErrorResponse
from utilities.api_models import SelectedAPI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGODB_URL = os.getenv("MONGODB_URL")
DATABASE_NAME = os.getenv("DATABASE_NAME")
CLOUDINARY_NAME = os.getenv("CLOUDINARY_NAME")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")

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

mongo_client = MongoClient(
    MONGODB_URL if MONGODB_URL else "mongodb://localhost:27017", server_api=ServerApi('1'))
db = mongo_client[DATABASE_NAME if DATABASE_NAME else "john-uk"]
collection = db["usage"]

try:
    mongo_client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

cloudinary.config(
    cloud_name=CLOUDINARY_NAME,
    api_key=CLOUDINARY_API_KEY,
    api_secret=CLOUDINARY_API_SECRET,
)


def upload_file_to_cloudinary(file, resource_type):
    upload_result = cloudinary.uploader.upload(
        file, folder="John-UK", resource_type=resource_type)
    return upload_result["secure_url"]


@app.post("/process")
async def process_files(text: str = Form(...), file: UploadFile = File(...), email: str = Form(...), selected_api: str = Form(...)):
    is_video = file.content_type.startswith('video')
    selected_api = selected_api.strip()
    text = text.strip()
    email = email.strip()

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

        if selected_api == "AWS_RECOGNITION":
            base64Frames = convert_video_to_images_for_aws(video_path, 100)
        else:
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

        if selected_api == "AWS_RECOGNITION":
            base64Frames.append(encode_image_for_aws(image_file_loc))
        else:
            base64Frames.append(encode_image(image_file_loc))
    else:
        response = ErrorResponse(statusCode=400,
                                 message="Invalid file type. Only images and videos are supported.")
        return JSONResponse(content=response.to_dict(), status_code=400)

    api_service = ApiServices()
    result = None
    success = False

    if selected_api == SelectedAPI.OPENAI.value:
        result, success = api_service.process_with_openai(
            text, base64Frames, is_video)

    elif selected_api == SelectedAPI.CLAUDE_ANTHROPIC.value:
        result, success = api_service.process_cloude_anthropic(
            text, base64Frames, is_video, "image/jpeg" if is_video else file.content_type)

    elif selected_api == SelectedAPI.AWS_RECOGNITION.value:
        result, success = api_service.process_aws_recognition(
            text, base64Frames, is_video)

    elif selected_api == SelectedAPI.GOOGLE_VISION.value:
        result, success = api_service.process_google_vission(
            text, base64Frames, is_video)

    elif selected_api == SelectedAPI.ASTICA.value:
        result, success = api_service.process_astrica_ai(
            text, base64Frames, is_video)

    else:
        result, success = {
            "status_code": 400,
            "message": "Invalid API selected!"
        }, False

    if not success:
        response = ErrorResponse(
            statusCode=result['status_code'], message=result['message'])
        return JSONResponse(content=response.to_dict(), status_code=result['status_code'])

    file_url = upload_file_to_cloudinary(
        video_path if is_video else image_file_loc, "video" if is_video else "image")

    current_time = datetime.datetime.now().isoformat()

    response = Response(statusCode=200, email=email, questionText=text, answerText=result['message'],
                        fileUrl=file_url, date=current_time, model=selected_api)

    collection.insert_one(response.to_dict())
    
    api_service.send_email(response, is_video)

    return JSONResponse(content=response.to_dict(), status_code=200)


@ app.get("/usage")
async def get_usage_data(email: str):
    try:
        usage_data = collection.find({"email": email})
        response = []
        for data in usage_data:
            data['_id'] = str(data['_id'])
            response.append(data)
        return response
    except Exception as e:
        return {"error": str(e)}
