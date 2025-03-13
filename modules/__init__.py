"""
Speech translation modules package.
"""

# Import event bus for communication between components
from modules.event_bus import event_bus

# Import configuration manager
from modules.config import config_manager, AppConfig

# Import legacy functions for backward compatibility
from modules.speech_to_text import transcribe_streaming_with_voice_activity
from modules.text_translation import translate_text
from modules.text_to_speech import (
    stream_translate_and_speak, 
    stream_speech_synthesis, 
    get_voice_for_language, 
    stream_translate_to_speech
)
from modules.pipeline import process_speech_to_translation

# Import new service classes
from modules.speech_to_text import SpeechToTextService
from modules.text_translation import TranslationService
from modules.text_to_speech import TextToSpeechService
from modules.pipeline import PipelineService

# Import initialization functions for services
from modules.speech_to_text import SpeechToTextService 
from modules.text_translation import init_translation_service
from modules.text_to_speech import init_tts_service
from modules.pipeline import init_pipeline_service

# Import metrics tracker - using absolute import
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
# Keep only the metrics import to avoid circular dependencies
from metrics import tracker as metrics_tracker 