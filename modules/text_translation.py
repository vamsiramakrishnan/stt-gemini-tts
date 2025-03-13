#!/usr/bin/env python3
"""
Text translation functionality for the Live Speech Translation App.
"""

import asyncio
import logging
import re
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Union, Callable
import time

from google.genai import types

# Keep only the metrics import to avoid circular dependencies
from metrics import tracker
from modules.event_bus import event_bus
from modules.config import config_manager

# Set up logging
logger = logging.getLogger(__name__)

class TranslationService:
    """
    Translation service that handles text translation using Gemini API.
    Uses event-driven architecture for loose coupling with other components.
    """
    
    def __init__(self, genai_client):
        """
        Initialize the translation service.
        
        Args:
            genai_client: The Gemini API client
        """
        self.genai_client = genai_client
        self.executor = ThreadPoolExecutor(max_workers=config_manager.config.max_translation_threads)
        self.running = False
        self._lock = threading.RLock()
        self.active_translations = {}
    
    def start(self):
        """Start the translation service."""
        self.running = True
        logger.info("Translation service started")
        
        # Subscribe to relevant events
        event_bus.subscribe("transcription", self._handle_transcription)
        event_bus.subscribe("translate_text", self._handle_translate_request)
        event_bus.subscribe("prewarm_translation", self._handle_prewarm_translation)
    
    def stop(self):
        """Stop the translation service."""
        self.running = False
        self.executor.shutdown(wait=False)
        logger.info("Translation service stopped")
    
    def _handle_transcription(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle transcription event to automatically translate new transcripts.
        
        Args:
            event_type: The event type
            data: Event data containing the transcript
        """
        if not self.running or "transcript" not in data:
            return
            
        transcript = data["transcript"]
        source_lang = data.get("language", config_manager.config.default_source_lang)
        player = data.get("player")  # Get player from the event data
        
        # Handle case where source_lang might be None
        if source_lang is None:
            # Use a hardcoded default if config doesn't have one
            if config_manager.config.default_source_lang is None:
                source_lang = "en"  # Fallback to English as default
                logger.info(f"No source language in config, using hardcoded default: {source_lang}")
            else:
                source_lang = config_manager.config.default_source_lang
                logger.info(f"No source language in event, using config default: {source_lang}")
            
        # Extract language code from source_lang if it's in format like "en-US"
        if "-" in source_lang:
            source_lang = source_lang.split("-")[0]
            
        # Get target language from config
        target_lang = config_manager.config.default_target_lang
        # Handle case where target_lang might be None
        if target_lang is None:
            logger.warning("No target language configured, defaulting to English")
            target_lang_code = "en"
        elif "-" in target_lang:
            # For translation we just need the language code without region
            target_lang_code = target_lang.split("-")[0]
        else:
            target_lang_code = target_lang
        
        # Start translation in a separate thread
        self._translate_text_async(
            transcript,
            source_lang,
            target_lang_code,
            stream_id=data.get("stream_id"),
            player=player  # Pass player to the async method
        )
    
    def _handle_translate_request(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle explicit translation request event.
        
        Args:
            event_type: The event type
            data: Event data containing text to translate
        """
        if not self.running or "text" not in data:
            return
            
        text = data["text"]
        source_lang = data.get("source_lang", config_manager.config.default_source_lang)
        target_lang = data.get("target_lang", config_manager.config.default_target_lang)
        player = data.get("player")  # Get player from the event data
        
        # Extract language codes
        if "-" in source_lang:
            source_lang = source_lang.split("-")[0]
        if "-" in target_lang:
            target_lang = target_lang.split("-")[0]
        
        # Start translation in a separate thread
        self._translate_text_async(
            text,
            source_lang,
            target_lang,
            callback_id=data.get("callback_id"),
            stream_id=data.get("stream_id"),
            player=player  # Pass player to the async method
        )
    
    def _handle_prewarm_translation(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle a prewarm request for the translation service.
        This creates a dummy translation request to initialize the service and reduce latency.
        
        Args:
            event_type: The event type
            data: Event data containing source_lang and target_lang
        """
        if not self.running:
            return
            
        source_lang = data.get("source_lang", config_manager.config.default_source_lang)
        target_lang = data.get("target_lang", config_manager.config.default_target_lang)
        
        if not source_lang or not target_lang:
            logger.warning("Cannot prewarm translation service without source and target languages")
            return
            
        logger.info(f"Prewarming translation service for {source_lang} to {target_lang}")
        
        # Create a simple text to translate
        text = "Hello, this is a test."
        
        # Submit a task to the executor to prewarm the service
        self.executor.submit(self._prewarm_worker, text, source_lang, target_lang)
    
    def _prewarm_worker(self, text: str, source_lang: str, target_lang: str) -> None:
        """
        Worker function that runs in a separate thread to prewarm the translation service.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
        """
        try:
            # Create a prompt for translation
            prompt = f"Translate the following {source_lang} text to {target_lang}. Only return the translation, nothing else: '{text}'"
            
            # Get translation config from config manager
            config = config_manager.config.translation_config
            
            logger.debug("Sending prewarm request to translation service")
            
            # Send a request to initialize the service
            self.genai_client.models.generate_content(
                model=config.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=config.temperature,
                    max_output_tokens=config.max_output_tokens,
                    top_p=config.top_p,
                    top_k=config.top_k
                )
            )
            
            logger.info("Translation service prewarmed successfully")
            
        except Exception as e:
            logger.error(f"Error prewarming translation service: {e}")
            import traceback
            traceback.print_exc()
    
    def _translate_text_async(self, text, source_lang, target_lang, callback_id=None, stream_id=None, player=None):
        """
        Submit translation task to thread pool.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            callback_id: Optional ID for callback correlation
            stream_id: Optional stream ID for correlation
            player: Optional audio player for output
        """
        if not text or not text.strip():
            # Emit empty translation event
            event_bus.publish("translation_result", {
                "text": text,
                "translation": "",
                "source_lang": source_lang,
                "target_lang": target_lang,
                "callback_id": callback_id,
                "stream_id": stream_id,
                "player": player  # Include player in the event
            })
            return
            
        # Generate a unique ID for this translation task
        translation_id = f"trans_{id(text)}_{source_lang}_{target_lang}"
        
        with self._lock:
            if translation_id in self.active_translations:
                # Instead of just logging a warning, check if the translation has the same parameters
                existing_translation = self.active_translations[translation_id]
                
                # If it's the same text and languages, just add the new stream_id and player to the existing task
                if (existing_translation["text"] == text and 
                    existing_translation["source_lang"] == source_lang and 
                    existing_translation["target_lang"] == target_lang):
                    
                    logger.info(f"Translation {translation_id} is already in progress, adding new stream_id and player")
                    
                    # Update the existing translation with the new stream_id and player if provided
                    if stream_id and not existing_translation.get("stream_id"):
                        existing_translation["stream_id"] = stream_id
                    
                    if player and not existing_translation.get("player"):
                        existing_translation["player"] = player
                        
                    if callback_id and not existing_translation.get("callback_id"):
                        existing_translation["callback_id"] = callback_id
                        
                    return
                else:
                    # If it's a different text or languages, log a warning and continue with a new ID
                    logger.warning(f"Translation {translation_id} is already in progress with different parameters")
                    translation_id = f"trans_{id(text)}_{source_lang}_{target_lang}_{time.time()}"
                
            self.active_translations[translation_id] = {
                "text": text,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "callback_id": callback_id,
                "stream_id": stream_id,
                "player": player  # Store player reference
            }
        
        # Submit translation task to executor
        self.executor.submit(
            self._translation_worker,
            translation_id,
            text,
            source_lang,
            target_lang,
            callback_id,
            stream_id,
            player  # Pass player to the worker
        )
        
        logger.debug(f"Submitted translation task {translation_id}")
    
    def _translation_worker(self, translation_id, text, source_lang, target_lang, callback_id, stream_id, player):
        """
        Worker function that runs in a separate thread to process translation.
        
        Args:
            translation_id: Unique ID for this translation task
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            callback_id: Optional ID for callback correlation
            stream_id: Optional stream ID for correlation
            player: Optional audio player for output
        """
        try:
            logger.info(f"Translating from {source_lang} to {target_lang}: '{text}'")
            
            prompt = f"Translate the following {source_lang} text to {target_lang}. Only return the translation, nothing else: '{text}'"
            logger.debug(f"Prompt: {prompt}")
            
            # Start timing Gemini
            tracker.start_timer("gemini")
            
            # Get translation config from config manager
            config = config_manager.config.translation_config
            
            # Collect translation chunks for streaming translation
            translation_chunks = []
            partial_translation = ""
            
            token_count = 0
            for chunk in self.genai_client.models.generate_content_stream(
                model=config.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=config.temperature,
                    max_output_tokens=config.max_output_tokens,
                    top_p=config.top_p,
                    top_k=config.top_k
                )
            ):
                if not self.running:
                    break
                    
                if chunk.text:
                    token_count += 1  # Count tokens
                    translation_chunks.append(chunk.text)
                    partial_translation += chunk.text
                    
                    # Determine if this chunk is meaningful (e.g., contains complete words)
                    is_meaningful = len(partial_translation.split()) >= 3
                    
                    # Emit partial translation event with more metadata
                    event_bus.publish("translation_partial", {
                        "text": text,
                        "partial_text": partial_translation,
                        "source_lang": source_lang,
                        "target_lang": target_lang,
                        "callback_id": callback_id,
                        "stream_id": stream_id,
                        "translation_id": translation_id,
                        "token_count": token_count,  # Include token count
                        "is_meaningful": is_meaningful,  # Explicit flag for meaningful chunks
                        "timestamp": time.time(),  # Add precise timestamp
                        "player": player
                    })
                    
                    logger.debug(f"Partial translation: {partial_translation}")
            
            # Stop timing Gemini
            tracker.stop_timer("gemini")
            
            translation = ''.join(translation_chunks).strip()
            logger.info(f"Translation result: '{translation}'")
            
            # Emit translation result event
            event_bus.publish("translation_result", {
                "text": text,
                "translation": translation,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "callback_id": callback_id,
                "stream_id": stream_id,
                "translation_id": translation_id,
                "player": player
            })
            
        except Exception as e:
            logger.error(f"Error in translation worker: {e}")
            import traceback
            traceback.print_exc()
            
            # Stop timing Gemini in case of error
            tracker.stop_timer("gemini")
            
            # Emit error event
            event_bus.publish("translation_error", {
                "text": text,
                "error": str(e),
                "source_lang": source_lang,
                "target_lang": target_lang,
                "callback_id": callback_id,
                "stream_id": stream_id,
                "translation_id": translation_id,
                "player": player
            })
        finally:
            # Clean up this translation task
            with self._lock:
                if translation_id in self.active_translations:
                    del self.active_translations[translation_id]

# Legacy function maintained for backward compatibility
def translate_text(genai_client, text, source_lang="en", target_lang="es"):
    """
    Legacy function that translates text using Gemini.
    Now implemented to use the event-based architecture.
    
    Args:
        genai_client: The Gemini client
        text: Text to translate
        source_lang: Source language code
        target_lang: Target language code
        
    Returns:
        Translated text
    """
    if not text.strip():
        return ""
    
    # Create an event to signal when translation is complete
    translation_complete = threading.Event()
    translation_result = [None]
    translation_error = [None]
    
    # Create event handlers
    def handle_translation_result(event_type, data):
        if data.get("callback_id") == callback_id:
            translation_result[0] = data.get("translation", "")
            translation_complete.set()
    
    def handle_translation_error(event_type, data):
        if data.get("callback_id") == callback_id:
            translation_error[0] = data.get("error", "Unknown error")
            translation_complete.set()
    
    # Generate a unique callback ID
    callback_id = f"translate_text_{id(text)}"
    
    # Subscribe to events
    unsubscribe_funcs = [
        event_bus.subscribe("translation_result", handle_translation_result),
        event_bus.subscribe("translation_error", handle_translation_error)
    ]
    
    try:
        # Start the translation service if it's not already running
        # This is a simple singleton pattern for the legacy function
        if not hasattr(translate_text, "_service"):
            translate_text._service = TranslationService(genai_client)
            translate_text._service.start()
        
        # Publish translation request
        event_bus.publish("translate_text", {
            "text": text,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "callback_id": callback_id
        })
        
        # Wait for translation to complete with timeout
        if not translation_complete.wait(timeout=30.0):
            logger.error("Translation timed out")
            return ""
        
        # Check for errors
        if translation_error[0]:
            logger.error(f"Translation error: {translation_error[0]}")
            return ""
        
        # Return translation
        return translation_result[0] or ""
        
    finally:
        # Unsubscribe from events
        for unsubscribe in unsubscribe_funcs:
            unsubscribe()

# Module initialization function
def init_translation_service(genai_client):
    """
    Initialize the translation service.
    
    Args:
        genai_client: The Gemini client
        
    Returns:
        The translation service instance
    """
    service = TranslationService(genai_client)
    service.start()
    return service 