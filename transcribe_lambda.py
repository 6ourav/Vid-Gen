import json
import boto3
import os
import re
import time 
import uuid
from urllib.parse import unquote_plus

s3 = boto3.client("s3")
transcribe = boto3.client("transcribe")

def lambda_handler(event, context):
    print("Lambda triggered with event:", json.dumps(event))  

    if "Records" not in event:
        print("No records found in event.")
        return {"statusCode": 400, "body": "Invalid event structure"}

    record = event["Records"][0]
    bucket_name = record["s3"]["bucket"]["name"]
    file_key = unquote_plus(record["s3"]["object"]["key"]) 

    print(f"Processing file: {file_key} from bucket: {bucket_name}")

    if not file_key.lower().endswith((".mp4", ".wav", ".flac", ".m4a")):
        print(f"Skipping non-audio/video file: {file_key}")
        return {"statusCode": 400, "body": "Not a valid media file"}

    sanitized_name = re.sub(r"[^a-zA-Z0-9-]", "-", file_key)[:100]
    timestamp = int(time.time())
    unique_id = uuid.uuid4().hex[:8]
    job_name = f"transcription-{sanitized_name}-{unique_id}-{timestamp}"

    print(f"Sanitized job name: {job_name}")

    try:
        response = transcribe.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={"MediaFileUri": f"s3://{bucket_name}/{file_key}"},
            MediaFormat=file_key.split(".")[-1],  
            LanguageCode="en-US",
            OutputBucketName="video-dummy-transcript-bucket"
        )
        print(f"Started transcription job: {job_name}")
        return {"statusCode": 200, "body": "Transcription job started"}
    except Exception as e:
        print(f"Error starting transcription: {str(e)}")
        return {"statusCode": 500, "body": "Transcription failed"}
