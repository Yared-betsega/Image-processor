import openai
from openai import OpenAI
from anthropic import Anthropic
import boto3
from google.cloud import vision
from google.oauth2 import service_account

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

from dotenv import load_dotenv
import requests
import json
import os

from utilities.response import Response


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLAUDE_ANTHROPIC_API_KEY = os.getenv("CLAUDE_ANTHROPIC_API_KEY")
ASTICA_API_KEY = os.getenv("ASTICA_API_KEY")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
PASSWORD = os.getenv("PASSWORD")


class ApiServices:
    def __init__(self) -> None:
        pass

    def process_with_openai(self, text, base64Frames, is_video):
        gap = len(base64Frames) // 45 if is_video else 1
        start = 25 if is_video else 0

        client = OpenAI(api_key=OPENAI_API_KEY)

        try:
            starter = "These are frames from a video that a user want me to process from CCTV. " if is_video else "This is an image a user want me to process. "
            PROMPT_MESSAGES = [
                {
                    "role": "user",
                    "content": [
                        starter + text + ", The answer must be formatted in HTML. Aslo Don't include any description outside the html. Since I will embed the html in other web app, don't put it inside html or body tags.",
                        *map(lambda x: {"image": x, "resize": 768},
                             base64Frames[start::gap]),
                    ],
                },
            ]

            params = {
                "model": "gpt-4-vision-preview",
                "messages": PROMPT_MESSAGES,
                "max_tokens": 200,
            }

            result = client.chat.completions.create(**params)
            return {
                "status_code": 200,
                "message": result.choices[0].message.content
            }, True

        except openai.BadRequestError as e:
            return {
                'status_code': e.status_code,
                'message': e.message
            }, False

        except Exception as e:
            return {
                'status_code': 400,
                'message': str(e)
            }, False

    def process_cloude_anthropic(self, text, base64Frames, is_video, media_type):
        gap = len(base64Frames) // 20 if is_video else 1
        start = 25 if is_video else 0

        client = Anthropic(api_key=CLAUDE_ANTHROPIC_API_KEY.strip())

        try:
            starter = "These are frames from a video that a user want me to process from CCTV. " if is_video else "This is an image a user want me to process. "
            last = ", The answer must be formatted in HTML. Aslo Don't include any description outside the html. Since I will embed the html in other web app, don't put it inside html or body tags."

            image_content = []
            image_counter = 1
            for x in base64Frames[start::gap]:
                image_content.append(
                    {
                        "type": "text",
                        "text": "Image " + str(image_counter) + ":"
                    },
                )
                image_content.append(
                    {
                        "type": "image",
                        "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": x,
                        }
                    },
                )
                image_counter += 1

            image_content.append(
                {
                    "type": "text",
                    "text": starter + text + last
                }
            )

            PROMPT_MESSAGES = [
                {
                    "role": "user",
                    "content": image_content,
                },
            ]

            params = {
                "model": "claude-3-opus-20240229",
                "max_tokens": 1024,
                "messages": PROMPT_MESSAGES,
            }

            result = client.messages.create(**params)
            return {
                'status_code': 200,
                'message': result.content[0].text
            }, True

        except Exception as e:
            return {
                'status_code': e.status_code,
                'message': e.message
            }, False

    def process_aws_recognition(self, text, base64Frames, is_video):
        client = boto3.client(
            'rekognition',
            region_name='us-east-1',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            aws_session_token=AWS_SESSION_TOKEN
        )

        response_labels = client.detect_labels(
            Image={'Bytes': base64Frames[0]})
        labels = [label['Name'] for label in response_labels['Labels']]

        response_text = client.detect_text(
            Image={'Bytes': base64Frames[0]})
        text = [word['DetectedText']
                for word in response_text['TextDetections']]

        description = self.generate_description_for_aws_recognition(
            labels, text)
        return description

    def process_google_vission(self, text, base64Frames, is_video):
        credentials = service_account.Credentials.from_service_account_file(
            './service-account-file.json')

        try:
            client = vision.ImageAnnotatorClient(credentials=credentials)
            image = vision.Image(content=base64Frames[0])

            analysis_results = {
                'labels': [],
                'objects': [],
                'texts': [],
                'faces': []
            }

            label_response = client.label_detection(image=image)
            for label in label_response.label_annotations:
                analysis_results['labels'].append({
                    'description': label.description,
                    'score': label.score * 100
                })

            object_response = client.object_localization(image=image)
            for obj in object_response.localized_object_annotations:
                analysis_results['objects'].append({
                    'name': obj.name,
                    'score': obj.score * 100,
                    'bounding_poly': [[{'x': vertex.x, 'y': vertex.y} for vertex in obj.bounding_poly.normalized_vertices]]
                })

            text_response = client.text_detection(image=image)
            for text in text_response.text_annotations:
                analysis_results['texts'].append({
                    'description': text.description
                })

            face_response = client.face_detection(image=image)
            emotions = ['joy', 'sorrow', 'anger', 'surprise']
            for index, face in enumerate(face_response.face_annotations, start=1):
                face_data = {'face': index, 'emotions': {}}
                for emotion in emotions:
                    likelihood = getattr(face, f'{emotion}_likelihood')
                    face_data['emotions'][emotion] = likelihood
                analysis_results['faces'].append(face_data)

            errors = [r.error.message for r in [label_response,
                                                object_response, text_response, face_response] if r.error.message]
            if errors:
                return {
                    'status_code': 400,
                    'message': 'API Errors:\n' + '\n'.join(errors)
                }, False

            result = self.convert_json_to_html_google_vision(analysis_results)
            return {
                'status_code': 200,
                'message': result
            }, True

        except Exception as e:
            return {
                'status_code': e.code,
                'message': e.message
            }, False

    def process_astrica_ai(self, text, base64Frames, is_video):
        asticaAPI_timeout = 100
        asticaAPI_endpoint = 'https://vision.astica.ai/describe'
        asticaAPI_modelVersion = '2.5_full'

        if 1 == 2:
            asticaAPI_input = 'https://astica.ai/example/asticaVision_sample.jpg'
        else:
            asticaAPI_input = base64Frames[0]

        asticaAPI_visionParams = ''
        starter = "These are frames from a video that a user want me to process from CCTV. " if is_video else "This is an image a user want me to process. "
        last = ", The answer must be formatted in HTML. Aslo Don't include any description outside the html. Since I will embed the html in other web app, don't put it inside html or body tags."
        asticaAPI_gpt_prompt = starter + text + last
        asticaAPI_prompt_length = 90

        # only used if visionParams includes "objects_custom" (v2.5_full or higher)
        asticaAPI_objects_custom_kw = ''

        asticaAPI_payload = {
            'tkn': ASTICA_API_KEY,
            'modelVersion': asticaAPI_modelVersion,
            'visionParams': asticaAPI_visionParams,
            'input': asticaAPI_input,
            'gpt_prompt': asticaAPI_gpt_prompt,
            'prompt_length': asticaAPI_prompt_length,
            'objects_custom_kw': asticaAPI_objects_custom_kw
        }

        def asticaAPI(endpoint, payload, timeout):
            response = requests.post(endpoint, data=json.dumps(
                payload), timeout=timeout, headers={'Content-Type': 'application/json', })
            if response.status_code == 200:
                return response.json()
            else:
                return {'status': 'error', 'error': 'Failed to connect to the API.'}

        try:
            asticaAPI_result = requests.post(asticaAPI_endpoint, data=json.dumps(
                asticaAPI_payload), timeout=asticaAPI_timeout, headers={'Content-Type': 'application/json', }).json()

            if 'error' in asticaAPI_result:
                return {
                    'status_code': 400,
                    'message': asticaAPI_result['error']
                }, False

            result = self.convert_json_to_html_astica(asticaAPI_result)
            return {
                'status_code': 200,
                'message': result
            }, True

        except Exception as e:
            return {
                'status_code': e.status_code,
                'message': e.message
            }, False

    def send_email(self, data: Response, is_video):

        subject = "Video " if is_video else "Image " + "Processing Result"
        answer = data.answerText
        if 'html' in answer:
            answer = answer[7:-3]

        question_data = f"""
                <p>Model Used: {data.model}</p>
                <p>Date: {data.date}</p>
                <p>Question: {data.questionText}</p>
            """
        if is_video:
            thumbnail = self.extract_thumbnail(data.fileUrl)
            video_html = f'Video: <a href="{data.fileUrl}"><img src="{thumbnail}" height="300px"/></a>'
        else:
            image_html = f'Image: <a href="{data.fileUrl}"><img src="{data.fileUrl}" height="300px"/></a>'
        file_data =  video_html if is_video else image_html
        
        answer_data = f"""
                <p> Answer: {answer} </p>
                <br >
                <p> Best Regards, </p>
                <p > John UK < /p >
        """
    
        body = question_data + file_data + answer_data

        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = data.email
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'html'))

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()  # Secure the connection

        server.login(SENDER_EMAIL, PASSWORD)

        text = msg.as_string()
        server.sendmail(SENDER_EMAIL, data.email, text)

        server.quit()

    def generate_description_for_aws_recognition(self, labels, text):
        description = "This image contains: "
        if labels:
            description += ', '.join(labels)
        if text:
            description += f" and the following text: {', '.join(text)}"
        return description

    def convert_json_to_html_google_vision(self, analysis_results):
        html_content = []
        # Begin with a base structure

        html_content.append('<h1>Image Analysis Results</h1>')

        # Labels Section
        if analysis_results.get('labels'):
            html_content.append('<h2>Detected Labels</h2>')
            html_content.append('<ul>')
            for label in analysis_results['labels']:
                html_content.append(f'<li>{label["description"]}: {label["score"]:.2f}% confidence</li>')
            html_content.append('</ul>')

        # Objects Section
        if analysis_results.get('objects'):
            html_content.append('<h2>Detected Objects</h2>')
            html_content.append('<ul>')
            for obj in analysis_results['objects']:
                html_content.append(f'<li>{obj["name"]}: {obj["score"]:.2f}% confidence</li>')
            html_content.append('</ul>')

        # Text Section
        if analysis_results.get('texts'):
            html_content.append('<h2>Detected Texts</h2>')
            html_content.append('<ul>')
            for text in analysis_results['texts']:
                html_content.append(f'<li>{text["description"]}</li>')
            html_content.append('</ul>')

        # Faces Section
        if analysis_results.get('faces'):
            html_content.append('<h2>Detected Faces and Emotions</h2>')
            for face in analysis_results['faces']:
                html_content.append(f'<h3>Face {face["face"]}</h3>')
                html_content.append('<ul>')
                for emotion, likelihood in face['emotions'].items():
                    html_content.append(f'<li>{emotion.capitalize()}: {likelihood}</li>')
                html_content.append('</ul>')

        # Combine the HTML parts into a single string
        html_result = "\n".join(html_content)
        return html_result

    def convert_json_to_html_astica(self, analysis_results):
        html_content = []
        # Begin with a base structure

        html_content.append('<h1>Image Analysis Results</h1>')

        # Labels Section
        if analysis_results.get('caption'):
            html_content.append('<h2>Description</h2>')
            html_content.append(f'<p>{analysis_results["caption_GPTS"]}</p>')

        # Objects Section
        if analysis_results.get('objects'):
            html_content.append('<h2>Detected Objects</h2>')
            html_content.append('<ul>')
            for obj in analysis_results['objects']:
                html_content.append(f'<li>{obj["name"]}: {obj["confidence"]:.2f}% confidence</li>')
            html_content.append('</ul>')

        if analysis_results.get('moderate'):
            html_content.append('<h2>Moderate</h2>')
            html_content.append('<ul>')
            html_content.append('<li>Is adult content?: ' + str(obj["moderate"]['isAdultContent']) + '</li>')
            html_content.append('<li>Is racy content?: ' + str(obj["moderate"]['isRacyContent']) + '</li>')
            html_content.append('<li>Is gory content?: ' + str(obj["moderate"]['isGoryContent']) + '</li>')

            html_content.append('</ul>')

        # Faces Section
        if analysis_results.get('faces'):
            html_content.append('<h2>Detected Faces and Emotions</h2>')
            for face in analysis_results['faces']:
                html_content.append('<h3>Face</h3>')
                html_content.append('<ul>')
                html_content.append('<li>Age: ' + str(face['age']) + '</li>')
                html_content.append('<li>Gender: ' + str(face['gender']) + '</li>')
                html_content.append('</ul>')

        # Combine the HTML parts into a single string
        html_result = "\n".join(html_content)
        return html_result
    
    def extract_thumbnail(self, video_url):
        base_url, _ = video_url.rsplit('.', 1)
        thumbnail_url = f"{base_url}.jpg"
        return thumbnail_url
