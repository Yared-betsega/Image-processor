import openai
from openai import OpenAI
from fastapi import FastAPI, UploadFile, File, Form
from typing import List
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import cloudinary
import cloudinary.uploader
import os
import shutil
from utils import convert_video_to_images, encode_frame, encode_image


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

# Connect to MongoDB
print(MONGODB_URL)
mongo_client = MongoClient(MONGODB_URL if MONGODB_URL else "mongodb://localhost:27017", server_api=ServerApi('1'))
db = mongo_client[DATABASE_NAME if DATABASE_NAME else "john-uk"]
collection = db["usage"]

try:
    mongo_client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

# Configure Cloudinary
cloudinary.config(
    cloud_name=CLOUDINARY_NAME,
    api_key=CLOUDINARY_API_KEY,
    api_secret=CLOUDINARY_API_SECRET,
)

# Function to upload file to Cloudinary and return public link
def upload_file_to_cloudinary(file, resource_type):
    upload_result = cloudinary.uploader.upload(file, folder="John-UK", resource_type = resource_type)
    return upload_result["secure_url"]

# Store information in MongoDB and upload file to Cloudinary
@app.post("/process")
async def process_files(text: str = Form(...), file: UploadFile = File(...), email: str = Form(...)):

    print("Entered")
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

        print("Sending request to openai")
        result = client.chat.completions.create(**params)
        

        print("Uploading file to cloudninary")
        # Upload file to Cloudinary
        file_url = upload_file_to_cloudinary(video_path if is_video else image_file_loc, "video" if is_video else "image")

        # Store information in MongoDB
        data = {
            "email": email,
            "questionText": text,
            "answerText": result.choices[0].message.content,
            "fileUrl" : file_url
        }
        print("Adding to database")
        collection.insert_one(data)

        print("Returning result")
        return result
    
    except openai.BadRequestError as e:
        return e.response.json()
    
    except Exception as e:
        return {"error": str(e)}
        



@app.get("/usage")
async def get_usage_data(email: str):
    print(email)
    try:
        usage_data = collection.find({"email": email})
        response = []
        for data in usage_data:
            data['_id'] = str(data['_id'])
            response.append(data)
        return response
    except Exception as e:
        return {"error": str(e)}