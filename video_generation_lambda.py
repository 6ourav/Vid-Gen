import spacy
import json
import boto3
from openai import OpenAI
from moviepy.editor import AudioFileClip, ImageClip, concatenate_videoclips
import requests
from transformers import pipeline
from botocore.exceptions import ClientError


polly = boto3.client("polly")
s3 = boto3.client("s3")
nlp = spacy.load("en_core_web_sm")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def get_secret():
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name="us-east-2"
    )
    try:
        get_secret_value_response = client.get_secret_value(
            SecretId="openai/secret"
        )
    except ClientError as e:
        raise e
    secret = get_secret_value_response['SecretString']
    return secret

client = OpenAI(api_key = get_secret())


def lambda_handler(event, context):
    record = event["Records"][0]
    bucket_name = record["s3"]["bucket"]["name"]
    file_key = record["s3"]["object"]["key"]

    response = s3.get_object(Bucket=bucket_name, Key=file_key)
    transcript_data = json.loads(response["Body"].read())

    full_text = transcript_data["transcript"]

    doc = nlp(full_text)
    chunks = []
    temp_chunk = []

    for sent in doc.sents:
        temp_chunk.append(sent.text)
        if len(temp_chunk) >= 2:
            chunks.append(" ".join(temp_chunk))
            temp_chunk = []

    if temp_chunk:
        chunks.append(" ".join(temp_chunk))

    data = []
    for i, chunk in enumerate(chunks):
        summary = summarizer(chunk, max_length=15, do_sample=False)[0]["generated_text"]
        image_filename = f"chunk_{i}.png"
        audio_filename = f"chunk_{i}.mp3"

        image_path = generate_image(summary, image_filename)
        audio_path = text_to_speech(chunk, audio_filename)

        data.append({
            "chunk_id": i,
            "original_text": chunk,
            "summary": summary,
            "image": image_path,
            "audio": audio_path
        })

    final_path = create_final_video(data)
    upload_to_s3(final_path, "video-dummy-output-bucket", f"output_{file_key}")


def generate_image(prompt, output_filename):
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1792x1024",
        quality="standard",
        n=1,
    )
    image_url = response.data[0].url
    img_data = requests.get(image_url).content
    output_path = f"/tmp/{output_filename}"
    with open(output_path, "wb") as f:
        f.write(img_data)
    return output_path


def text_to_speech(text, output_filename):
    response = polly.synthesize_speech(
        Text=text, OutputFormat="mp3", VoiceId="Joanna"
    )
    output_path = f"/tmp/{output_filename}"
    with open(output_path, "wb") as file:
        file.write(response["AudioStream"].read())
    return output_path


def create_final_video(data, output_filename="final_video.mp4"):
    clips = []

    for item in data:
        audio_path = item["audio"]   
        image_path = item["image"]   

        audio_clip = AudioFileClip(audio_path)
        image_clip = ImageClip(image_path).set_duration(audio_clip.duration)

        video_clip = image_clip.set_audio(audio_clip)
        clips.append(video_clip)

    final_clip = concatenate_videoclips(clips, method="compose")
    final_path = "/tmp/" + output_filename
    final_clip.write_videofile(final_path, fps=24)

    return final_path


def upload_to_s3(file_path, bucket_name, key):
    s3 = boto3.client("s3")
    with open(file_path, "rb") as f:
        s3.upload_fileobj(f, bucket_name, key)


def get_secret():
    secret_name = "openai/secret"
    region_name = "us-east-2"

    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        raise e

    secret = get_secret_value_response['SecretString']
    return secret