import os
import boto3
from dotenv import load_dotenv

load_dotenv()

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

os.makedirs("data", exist_ok=True)

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION,
)

def download_pdfs():
    print(f"Checking bucket: {BUCKET_NAME}")
    objects = s3.list_objects_v2(Bucket=BUCKET_NAME)
    
    if "Contents" not in objects:
        print("No files found in bucket.")
        return
    
    for obj in objects["Contents"]:
        key = obj["Key"]
        if key.endswith(".pdf"):
            local_path = os.path.join("data", os.path.basename(key))
            print(f"Downloading {key} to {local_path}")
            s3.download_file(BUCKET_NAME, key, local_path)

if __name__ == "__main__":
    download_pdfs()
