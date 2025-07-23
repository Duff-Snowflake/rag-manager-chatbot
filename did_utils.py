import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DID_API_KEY = os.getenv("DID_API_KEY")
DID_BASE_URL = "https://api.d-id.com"

def generate_did_video(script_text, voice_id="en-US-Wavenet-F"):
    """
    Sends a request to D-ID API to generate a talking head video.
    You must have a D-ID API key set in your environment.
    """

    if not DID_API_KEY:
        raise ValueError("❌ DID_API_KEY not set. Add it to your .env or Streamlit Secrets.")

    headers = {
        "Authorization": f"Bearer {DID_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "source_url": "https://create-images-results.d-id.com/DefaultPresentationImage.png",  # Default avatar
        "script": {
            "type": "text",
            "input": script_text,
            "provider": {
                "type": "microsoft",
                "voice_id": voice_id
            },
            "ssml": False
        },
        "config": {
            "fluent": True,
            "pad_audio": 0.3
        }
    }

    # Create the video
    response = requests.post(f"{DID_BASE_URL}/talks", headers=headers, json=payload)

    if response.status_code != 200:
        raise Exception(f"❌ D-ID request failed: {response.status_code} - {response.text}")

    video_id = response.json().get("id")
    return f"https://studio.d-id.com/share/{video_id}"
