"""
Configuration for the speech-to-speech translation app.
"""
import os
import time
import pyaudio
from google.cloud.speech_v2 import SpeechClient
from google import genai
from google.cloud import texttospeech

# Audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Audio sampling rate
CHUNK = int(RATE / 10)  # 100ms chunks (more responsive than 1024)
PLAYBACK_RATE = 24000  # May differ from input rate based on TTS config
STREAMING_LIMIT = 60 * 1000  # 1 minute in milliseconds

# Project settings
PROJECT_ID = "vital-octagon-19612"
DEFAULT_SOURCE_LANG = "en-US"
DEFAULT_TARGET_LANG = "hi-IN"

def get_project_id():
    """Get the Google Cloud project ID from environment or config."""
    # First check environment variable
    env_project_id = os.environ.get('GOOGLE_CLOUD_PROJECT')
    if env_project_id:
        return env_project_id
    
    # Otherwise return the configured project ID
    return PROJECT_ID

def get_current_time():
    """Return current time in milliseconds."""
    return int(time.time() * 1000)

def setup_clients():
    """
    Initialize all the Google API clients needed for the application.
    Returns tuple of (speech_client, genai_client, tts_client)
    """
    print("Setting up Google Cloud clients...")
    
    # Initialize Speech-to-Text client
    try:
        speech_client = SpeechClient()
        print("Speech-to-Text client initialized.")
    except Exception as e:
        print(f"Error initializing Speech-to-Text client: {e}")
        speech_client = None
    
    # Initialize Gemini client for translation
    try:
        genai_client = genai.Client(vertexai=True, project=get_project_id(), location='us-central1')
        print("Gemini client initialized.")
    except Exception as e:
        print(f"Error initializing Gemini client: {e}")
        genai_client = None
    
    # Initialize Text-to-Speech client
    try:
        tts_client = texttospeech.TextToSpeechClient()
        print("Text-to-Speech client initialized.")
    except Exception as e:
        print(f"Error initializing Text-to-Speech client: {e}")
        tts_client = None
    
    return speech_client, genai_client, tts_client 