#!/usr/bin/env python3
"""
Text-to-speech functionality for the Live Speech Translation App.
"""

import asyncio
import logging
import re
import threading
import queue
import itertools
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Union
import time

from google.cloud import texttospeech
from google.genai import types

# Keep only the metrics import to avoid circular dependencies
from metrics import tracker
from modules.event_bus import event_bus
from modules.config import config_manager

# Set up logging
logger = logging.getLogger(__name__)

# Global TTS service instance
_tts_service_instance = None
_tts_service_lock = threading.RLock()

class TextToSpeechService:
    """
    Text-to-speech service that handles speech synthesis with streaming support.
    Uses event-driven architecture for loose coupling with other components.
    """
    
    @staticmethod
    def get_instance(tts_client=None):
        """
        Get or create the singleton instance of the TextToSpeechService.
        
        Args:
            tts_client: The Text-to-Speech client
            
        Returns:
            The TextToSpeechService instance
        """
        global _tts_service_instance, _tts_service_lock
        
        with _tts_service_lock:
            if _tts_service_instance is None and tts_client is not None:
                _tts_service_instance = TextToSpeechService(tts_client)
                _tts_service_instance.start()
            elif _tts_service_instance is not None and tts_client is not None:
                # If the instance exists but with a different client, restart it
                if _tts_service_instance.tts_client != tts_client:
                    _tts_service_instance.stop()
                    _tts_service_instance = TextToSpeechService(tts_client)
                    _tts_service_instance.start()
            
            return _tts_service_instance
    
    def __init__(self, tts_client):
        """
        Initialize the text-to-speech service.
        
        Args:
            tts_client: The Text-to-Speech client
        """
        self.tts_client = tts_client
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.running = False
        self._lock = threading.RLock()
        self.active_synthesizers = {}  # Keep the original name to avoid breaking existing code
        self.voice_cache = {}  # Cache for voice names by language
        self._subscribed = False  # Keep track of subscription status
    
    def start(self):
        """Start the text-to-speech service."""
        if self.running:
            logger.debug("TTS service is already running, skipping start")
            return
            
        self.running = True
        logger.info("Text-to-speech service started")
        
        # Only subscribe once to prevent duplicate event handling
        if not self._subscribed:
            # Subscribe to relevant events
            event_bus.subscribe("translation_result", self._handle_translation_result)
            event_bus.subscribe("synthesize_speech", self._handle_synthesize_request)
            event_bus.subscribe("prewarm_tts", self._handle_prewarm_tts)
            self._subscribed = True
    
    def stop(self):
        """Stop the text-to-speech service."""
        self.running = False
        self.executor.shutdown(wait=False)
        logger.info("Text-to-speech service stopped")
    
    def get_voice_for_language(self, target_lang):
        """
        Get the appropriate voice for the target language.
        
        Args:
            target_lang: The target language code
            
        Returns:
            The appropriate voice name for the language
        """
        voice_map = config_manager.config.tts_config.voice_map
        default_voice = config_manager.config.tts_config.default_voice
        
        voice_name = voice_map.get(target_lang, default_voice)
        logger.debug(f"Selected voice {voice_name} for language {target_lang}")
        return voice_name
    
    def _handle_translation_result(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle translation result event to automatically synthesize speech.
        
        Args:
            event_type: The event type
            data: Event data containing the translation
        """
        if not self.running or "translation" not in data:
            return
            
        translation = data["translation"]
        target_lang = data.get("target_lang", "")
        
        logger.debug(f"Translation result received for target language: '{target_lang}', text: '{translation}'")
        
        # Convert language code to TTS format if needed (e.g., "es" to "es-ES")
        if target_lang and "-" not in target_lang:
            # Map common language codes to default regions
            lang_region_map = {
                "en": "en-US",
                "es": "es-ES",
                "fr": "fr-FR",
                "de": "de-DE",
                "it": "it-IT",
                "ja": "ja-JP",
                "hi": "hi-IN"
            }
            target_lang = lang_region_map.get(target_lang, f"{target_lang}-{target_lang.upper()}")
            logger.debug(f"Mapped language code to: {target_lang}")
        
        # Get target language from config if not specified
        if not target_lang:
            target_lang = config_manager.config.default_target_lang
            logger.debug(f"Using default target language: {target_lang}")
        
        # Start speech synthesis in a separate thread
        logger.debug(f"Starting speech synthesis for language {target_lang} with player: {data.get('player') is not None}")
        self._synthesize_speech_async(
            translation,
            target_lang,
            callback_id=data.get("callback_id"),
            stream_id=data.get("stream_id"),
            player=data.get("player")
        )
    
    def _handle_synthesize_request(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle synthesize speech request event.
        
        Args:
            event_type: The event type
            data: Event data
        """
        if not self.running:
            return
            
        text = data.get("text")
        target_lang = data.get("target_lang")
        callback_id = data.get("callback_id")
        stream_id = data.get("stream_id")  # Make sure we get the stream_id
        player = data.get("player")
        
        if not text or not target_lang:
            logger.warning("Missing text or target_lang in synthesize speech request")
            return
            
        # Log the request for debugging
        logger.debug(f"Handling synthesize request: text='{text[:30]}...', target_lang={target_lang}, stream_id={stream_id}")
        
        # Submit the synthesis task to the executor
        self._synthesize_speech_async(text, target_lang, callback_id, stream_id, player)
    
    def _synthesize_speech_async(self, text, target_lang, callback_id=None, stream_id=None, player=None):
        """
        Submit speech synthesis task to thread pool.
        
        Args:
            text: Text to synthesize
            target_lang: Target language code
            callback_id: Optional ID for callback correlation
            stream_id: Optional stream ID for correlation
            player: Optional audio player for output
        """
        if not text or not text.strip():
            # Emit empty synthesis event
            event_bus.publish("synthesis_complete", {
                "text": text,
                "target_lang": target_lang,
                "callback_id": callback_id,
                "stream_id": stream_id
            })
            return
            
        # Generate a unique ID for this synthesis task
        synthesis_id = f"tts_{id(text)}_{target_lang}"
        
        with self._lock:
            if synthesis_id in self.active_synthesizers:
                logger.warning(f"Synthesis {synthesis_id} is already in progress")
                return
                
            self.active_synthesizers[synthesis_id] = {
                "text": text,
                "target_lang": target_lang,
                "callback_id": callback_id,
                "stream_id": stream_id,
                "player": player
            }
        
        # Announce TTS is starting
        event_bus.publish("tts_active", {
            "text": text,
            "target_lang": target_lang,
            "synthesis_id": synthesis_id
        })
        
        # Submit synthesis task to executor
        self.executor.submit(
            self._synthesis_worker,
            synthesis_id,
            text,
            target_lang,
            callback_id,
            stream_id,
            player
        )
        
        logger.debug(f"Submitted speech synthesis task {synthesis_id}")
    
    def _synthesis_worker(self, synthesis_id, text, target_lang, callback_id, stream_id, player):
        """
        Worker function that runs in a separate thread to process speech synthesis.
        
        Args:
            synthesis_id: Unique ID for this synthesis task
            text: Text to synthesize
            target_lang: Target language code
            callback_id: Optional ID for callback correlation
            stream_id: Optional stream ID for correlation
            player: Optional audio player for output
        """
        try:
            logger.info(f"Synthesizing speech in {target_lang}: '{text}'")
            
            voice_name = self.get_voice_for_language(target_lang)
            
            # Safely generate language_code from target_lang
            try:
                if not target_lang:
                    language_code = "en-US"  # Default fallback
                    logger.warning(f"Empty target_lang, defaulting to {language_code}")
                elif "-" not in target_lang:
                    # Handle case where target_lang doesn't contain a region
                    language_map = {
                        "en": "en-US",
                        "es": "es-ES",
                        "fr": "fr-FR",
                        "de": "de-DE",
                        "it": "it-IT",
                        "ja": "ja-JP",
                        "hi": "hi-IN"
                    }
                    language_code = language_map.get(target_lang, f"{target_lang}-{target_lang.upper()}")
                    logger.debug(f"No region in target_lang, generated language_code: {language_code}")
                else:
                    parts = target_lang.split("-")
                    language_code = f"{parts[0]}-{parts[1]}"
                    logger.debug(f"Generated language_code: {language_code}")
            except Exception as e:
                language_code = "en-US"  # Default fallback
                logger.error(f"Error generating language_code from {target_lang}: {e}, defaulting to {language_code}")
            
            logger.debug(f"Using voice {voice_name} with language code {language_code}")
            
            # Get TTS config from config manager
            tts_config = config_manager.config.tts_config
            
            # Start timing TTS
            tracker.start_timer("tts")
            
            # Try streaming synthesis first
            try:
                # Try with all parameters first
                try:
                    # Configure streaming TTS request with all parameters
                    streaming_config = texttospeech.StreamingSynthesizeConfig(
                        voice=texttospeech.VoiceSelectionParams(
                            name=voice_name,
                            language_code=language_code
                        )
                        # Note: speaking_rate, pitch, and volume_gain_db are not supported in StreamingSynthesizeConfig
                        # according to the Google Cloud docs
                    )
                except ValueError as e:
                    # Fallback to minimal configuration if extended parameters aren't supported
                    logger.warning(f"Falling back to basic configuration: {e}")
                    streaming_config = texttospeech.StreamingSynthesizeConfig(
                        voice=texttospeech.VoiceSelectionParams(
                            name=voice_name,
                            language_code=language_code
                        )
                    )
                
                # Create the config request
                config_request = texttospeech.StreamingSynthesizeRequest(streaming_config=streaming_config)
                
                # Create a request generator for the text
                def request_generator():
                    # Only use the input parameter as shown in the reference
                    yield texttospeech.StreamingSynthesizeRequest(
                        input=texttospeech.StreamingSynthesisInput(text=text)
                    )
                
                # Chain the config request with the text request
                streaming_responses = self.tts_client.streaming_synthesize(
                    itertools.chain([config_request], request_generator())
                )
                
                # Process the responses
                first_chunk = True
                for response in streaming_responses:
                    if not self.running:
                        break
                    
                    logger.debug(f"Received streaming audio chunk of size: {len(response.audio_content) if response.audio_content else 0} bytes")
                        
                    # Record the exact time when audio is sent to player
                    generation_time = time.time()
                    
                    if player:
                        playback_time = time.time()
                        player.play(response.audio_content)
                        
                        # Emit event for metrics tracking with explicit flags
                        event_bus.publish("tts_audio_chunk", {
                            "audio_content": None,  # Don't include audio content
                            "callback_id": callback_id,
                            "stream_id": stream_id,
                            "synthesis_id": synthesis_id,
                            "playback_time": playback_time,
                            "generation_time": generation_time,
                            "is_first_chunk": first_chunk,  # Flag for first chunk
                            "metrics_only": True
                        })
                    else:
                        logger.warning("No player provided for audio playback")
                        # Still emit the event for metrics tracking
                        event_bus.publish("tts_audio_chunk", {
                            "audio_content": response.audio_content,
                            "callback_id": callback_id,
                            "stream_id": stream_id,  # Ensure stream_id is always included
                            "synthesis_id": synthesis_id,
                            "generation_time": generation_time
                        })
                    
                    first_chunk = False  # No longer the first chunk
            except Exception as e:
                # If streaming synthesis fails, try standard synthesis as a fallback
                logger.warning(f"Streaming synthesis failed, falling back to standard synthesis: {e}")
                
                # Use the standard synthesize method which has a more stable API
                input_text = texttospeech.SynthesisInput(text=text)
                voice = texttospeech.VoiceSelectionParams(
                    name=voice_name, 
                    language_code=language_code
                )
                audio_config = texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                    speaking_rate=tts_config.speaking_rate,
                    pitch=tts_config.pitch,
                    volume_gain_db=tts_config.volume_gain_db
                )
                
                # Perform standard synthesis
                response = self.tts_client.synthesize_speech(
                    input=input_text,
                    voice=voice,
                    audio_config=audio_config
                )
                
                logger.debug(f"Received standard synthesis audio of size: {len(response.audio_content) if response.audio_content else 0} bytes")
                
                # Play or emit the single audio chunk
                if player:
                    logger.debug(f"Playing audio with player: {player}")
                    player.play(response.audio_content)
                else:
                    logger.warning("No player provided for audio playback")
                
                # Emit audio chunk event - ONLY emit if player is NOT provided
                # This prevents double-playback when both direct playback and event-based playback occur
                if not player:
                    event_bus.publish("tts_audio_chunk", {
                        "audio_content": response.audio_content,
                        "callback_id": callback_id,
                        "stream_id": stream_id,
                        "synthesis_id": synthesis_id
                    })
            
            # Stop timing TTS
            tracker.stop_timer("tts")
            
            # Emit synthesis complete event
            event_bus.publish("synthesis_complete", {
                "text": text,
                "target_lang": target_lang,
                "callback_id": callback_id,
                "stream_id": stream_id,
                "synthesis_id": synthesis_id
            })
            
            # Announce TTS is no longer active
            event_bus.publish("tts_inactive", {
                "text": text,
                "target_lang": target_lang,
                "synthesis_id": synthesis_id
            })
            
        except Exception as e:
            logger.error(f"Error in synthesis worker: {e}")
            import traceback
            traceback.print_exc()
            
            # Stop timing TTS in case of error
            tracker.stop_timer("tts")
            
            # Emit error event
            event_bus.publish("synthesis_error", {
                "text": text,
                "error": str(e),
                "target_lang": target_lang,
                "callback_id": callback_id,
                "stream_id": stream_id,
                "synthesis_id": synthesis_id
            })
            
            # Announce TTS is no longer active
            event_bus.publish("tts_inactive", {
                "text": text,
                "target_lang": target_lang,
                "synthesis_id": synthesis_id
            })
        finally:
            # Clean up this synthesis task
            with self._lock:
                if synthesis_id in self.active_synthesizers:
                    del self.active_synthesizers[synthesis_id]
    
    def _handle_prewarm_tts(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle a prewarm request for the TTS service.
        This creates a dummy synthesis request to initialize the service and reduce latency.
        
        Args:
            event_type: The event type
            data: Event data containing target_lang
        """
        if not self.running:
            return
            
        target_lang = data.get("target_lang", config_manager.config.default_target_lang)
        
        if not target_lang:
            logger.warning("Cannot prewarm TTS service without target language")
            return
            
        logger.info(f"Prewarming TTS service for language {target_lang}")
        
        # Create a simple text to synthesize
        text = "Hello, this is a test."
        
        # Submit a task to the executor to prewarm the service
        self.executor.submit(self._prewarm_worker, text, target_lang)
    
    def _prewarm_worker(self, text: str, target_lang: str) -> None:
        """
        Worker function that runs in a separate thread to prewarm the TTS service.
        
        Args:
            text: Text to synthesize
            target_lang: Target language code
        """
        try:
            # Get the voice for the target language
            voice = self.get_voice_for_language(target_lang)
            
            # Create a synthesis input
            synthesis_input = texttospeech.SynthesisInput(text=text)
            
            # Set the voice parameters
            voice_params = texttospeech.VoiceSelectionParams(
                language_code=target_lang,
                name=voice
            )
            
            # Set the audio config
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                sample_rate_hertz=24000
            )
            
            logger.debug("Sending prewarm request to TTS service")
            
            # Send a request to initialize the service
            self.tts_client.synthesize_speech(
                input=synthesis_input,
                voice=voice_params,
                audio_config=audio_config
            )
            
            logger.info("TTS service prewarmed successfully")
            
        except Exception as e:
            logger.error(f"Error prewarming TTS service: {e}")
            import traceback
            traceback.print_exc()

# Legacy functions maintained for backward compatibility
def get_voice_for_language(target_lang):
    """
    Get the appropriate voice for the target language.
    Legacy function that uses the config settings.
    
    Args:
        target_lang: The target language code
        
    Returns:
        The appropriate voice name for the language
    """
    voice_map = config_manager.config.tts_config.voice_map
    default_voice = config_manager.config.tts_config.default_voice
    
    voice_name = voice_map.get(target_lang, default_voice)
    return voice_name

def stream_speech_synthesis(tts_client, text, target_lang="es-US", player=None):
    """
    Legacy function that streams speech synthesis for the given text without translation.
    Now implemented to use the event-based architecture.
    
    Args:
        tts_client: The Text-to-Speech client
        text: The text to synthesize
        target_lang: The target language code
        player: The audio player for output
        
    Returns:
        None
    """
    if not text or not text.strip():
        return
    
    # Create an event to signal when synthesis is complete
    synthesis_complete = threading.Event()
    
    # Create event handler
    def handle_synthesis_complete(event_type, data):
        if data.get("callback_id") == callback_id:
            synthesis_complete.set()
    
    # Generate a unique callback ID
    callback_id = f"speech_synthesis_{id(text)}"
    
    # Subscribe to event
    unsubscribe_func = event_bus.subscribe("synthesis_complete", handle_synthesis_complete)
    
    try:
        # Get the singleton TTS service instance and ensure it's started
        service = TextToSpeechService.get_instance(tts_client)
        if not service.running:
            service.start()
        
        # Publish synthesis request
        event_bus.publish("synthesize_speech", {
            "text": text,
            "target_lang": target_lang,
            "callback_id": callback_id,
            "player": player
        })
        
        # Wait for synthesis to complete with timeout
        synthesis_complete.wait(timeout=30.0)
        
    finally:
        # Unsubscribe from event
        unsubscribe_func()

def stream_translate_and_speak(genai_client, tts_client, text, source_lang="en", target_lang="es-US", player=None):
    """
    Legacy function that translates text and streams the translation to speech synthesis.
    Now implemented to use the event-based architecture.
    
    Args:
        genai_client: The Gemini client for translation
        tts_client: The Text-to-Speech client
        text: The text to translate
        source_lang: The source language code
        target_lang: The target language code
        player: The audio player for output
        
    Returns:
        The translation as a string
    """
    from modules.text_translation import translate_text
    
    # First translate the text
    translation = translate_text(genai_client, text, source_lang, target_lang.split("-")[0])
    
    # Then synthesize the speech
    if translation:
        stream_speech_synthesis(tts_client, translation, target_lang, player)
    
    return translation

def stream_translate_to_speech(tts_client, text, player=None, target_lang="es-US", source_lang=None, pipeline_id=None):
    """
    Stream text directly to speech synthesis, without translation.
    Uses event-based architecture for coordination.
    
    Args:
        tts_client: The Text-to-Speech client
        text: The text to synthesize
        player: The audio player for output
        target_lang: The target language code
        source_lang: Optional source language (for metadata only)
        pipeline_id: Optional pipeline ID for event routing
        
    Returns:
        None
    """
    if not text.strip() or not player:
        return
        
    # Use the singleton TTS service
    tts_service = TextToSpeechService.get_instance(tts_client)
    
    # Generate a synthesis ID
    synthesis_id = f"synthesis_{id(text)}"
    
    # Create events for coordination
    synthesis_complete = threading.Event()
    
    # Create event handlers
    def handle_synthesis_complete(event_type, data):
        if data.get("synthesis_id") == synthesis_id:
            synthesis_complete.set()
    
    # Subscribe to events
    unsubscribe_func = event_bus.subscribe("synthesis_complete", handle_synthesis_complete)
    
    try:
        # Start the synthesis
        event_bus.publish("synthesize_speech", {
            "text": text,
            "target_lang": target_lang,
            "synthesis_id": synthesis_id,
            "player": player,
            "pipeline_id": pipeline_id  # Pass pipeline_id for better event routing
        })
        
        # Wait for synthesis to complete with timeout
        synthesis_complete.wait(timeout=30.0)
    finally:
        # Unsubscribe from events
        unsubscribe_func()

# Module initialization function
def init_tts_service(tts_client):
    """
    Initialize the text-to-speech service.
    
    Args:
        tts_client: The Text-to-Speech client
        
    Returns:
        The text-to-speech service instance
    """
    service = TextToSpeechService.get_instance(tts_client)
    service.start()
    return service 