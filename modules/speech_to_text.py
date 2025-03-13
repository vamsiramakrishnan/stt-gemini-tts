#!/usr/bin/env python3
"""
Speech-to-text functionality for the Live Speech Translation App.
"""

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Generator, List, Optional, Union
import time

from google.cloud.speech_v2.types import cloud_speech

# Keep only the metrics import to avoid circular dependencies
from metrics import tracker
from modules.event_bus import event_bus
from modules.config import config_manager

# Set up logging
logger = logging.getLogger(__name__)

class SpeechToTextService:
    """
    Speech-to-text service that handles streaming transcription with voice activity detection.
    Uses event-driven architecture to decouple from other components.
    """
    
    def __init__(self, speech_client):
        """
        Initialize the speech-to-text service.
        
        Args:
            speech_client: The Google Cloud Speech-to-Text v2 client
        """
        self.speech_client = speech_client
        self.active_streams = {}
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.running = False
        self._lock = threading.RLock()
    
    def start(self):
        """Start the speech-to-text service."""
        self.running = True
        logger.info("Speech-to-text service started")
        
        # Subscribe to relevant events
        event_bus.subscribe("audio_stream_created", self._handle_audio_stream)
        event_bus.subscribe("tts_active", self._handle_tts_active)
        event_bus.subscribe("tts_inactive", self._handle_tts_inactive)
        event_bus.subscribe("prewarm_stt", self._handle_prewarm_stt)
        
    def stop(self):
        """Stop the speech-to-text service."""
        self.running = False
        # Stop all active streams
        with self._lock:
            for stream_id, stream_data in list(self.active_streams.items()):
                self._stop_stream(stream_id)
        self.executor.shutdown(wait=False)
        logger.info("Speech-to-text service stopped")
    
    def _handle_audio_stream(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle a new audio stream event.
        
        Args:
            event_type: The event type
            data: Event data containing the audio stream
        """
        if not self.running or not data or "stream" not in data:
            return
            
        audio_stream = data["stream"]
        stream_id = data.get("stream_id", id(audio_stream))
        language_code = data.get("language_code", config_manager.config.stt_config.language_code)
        project_id = data.get("project_id", config_manager.config.stt_config.project_id)
        player = data.get("player")  # Get player from the event data
        
        self._start_transcription(audio_stream, stream_id, project_id, language_code, player)
    
    def _handle_tts_active(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle TTS active event to implement echo cancellation.
        
        Args:
            event_type: The event type
            data: Event data
        """
        # Optionally pause or modify STT processing during TTS
        logger.debug("TTS active, adjusting speech recognition sensitivity")
        
        # If echo cancellation is enabled, we could modify the VAD sensitivity
        # or mark streams as temporarily ignoring speech activity
        if config_manager.config.echo_cancellation_enabled:
            with self._lock:
                for stream_id in self.active_streams:
                    self.active_streams[stream_id]["echo_cancellation_active"] = True
    
    def _handle_tts_inactive(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle TTS inactive event.
        
        Args:
            event_type: The event type
            data: Event data
        """
        logger.debug("TTS inactive, restoring speech recognition sensitivity")
        
        # If echo cancellation is enabled, restore normal operation
        if config_manager.config.echo_cancellation_enabled:
            with self._lock:
                for stream_id in self.active_streams:
                    self.active_streams[stream_id]["echo_cancellation_active"] = False
    
    def _handle_prewarm_stt(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle a prewarm request for the STT service.
        This creates a dummy recognition request to initialize the service and reduce latency.
        
        Args:
            event_type: The event type
            data: Event data containing project_id and language_code
        """
        if not self.running:
            return
            
        project_id = data.get("project_id")
        language_code = data.get("language_code", config_manager.config.stt_config.language_code)
        
        if not project_id:
            logger.warning("Cannot prewarm STT service without project_id")
            return
            
        logger.info(f"Prewarming STT service with language {language_code}")
        
        # Submit a task to the executor to prewarm the service
        self.executor.submit(self._prewarm_worker, project_id, language_code)
    
    def _prewarm_worker(self, project_id: str, language_code: str) -> None:
        """
        Worker function that runs in a separate thread to prewarm the STT service.
        
        Args:
            project_id: Google Cloud project ID
            language_code: Language code for transcription
        """
        try:
            # Create a minimal recognition config
            recognition_config = cloud_speech.RecognitionConfig(
                explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                    encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=16000,
                    audio_channel_count=1
                ),
                language_codes=[language_code],
                model="latest_short",
            )
            
            # Create a recognizer path
            recognizer_path = f"projects/{project_id}/locations/global/recognizers/_"
            
            # Create a minimal audio content (silent audio)
            silent_audio = bytes([0] * 1600)  # 100ms of silence at 16kHz
            
            # Create a recognition request
            request = cloud_speech.RecognizeRequest(
                recognizer=recognizer_path,
                config=recognition_config,
                content=silent_audio
            )
            
            # Send the request to initialize the service
            logger.debug("Sending prewarm request to STT service")
            self.speech_client.recognize(request)
            logger.info("STT service prewarmed successfully")
            
        except Exception as e:
            logger.error(f"Error prewarming STT service: {e}")
            import traceback
            traceback.print_exc()
    
    def _start_transcription(self, audio_stream, stream_id, project_id, language_code, player=None):
        """
        Start transcription for an audio stream.
        
        Args:
            audio_stream: The audio stream to transcribe
            stream_id: Unique identifier for this stream
            project_id: Google Cloud project ID
            language_code: Language code for transcription
            player: Optional audio player for output
        """
        with self._lock:
            if stream_id in self.active_streams:
                logger.warning(f"Stream {stream_id} is already being processed")
                return
                
            self.active_streams[stream_id] = {
                "stream": audio_stream,
                "echo_cancellation_active": False,
                "player": player  # Store player reference
            }
        
        # Start transcription in a separate thread
        self.executor.submit(
            self._transcription_worker,
            audio_stream, 
            stream_id,
            project_id, 
            language_code,
            player  # Pass player to the worker
        )
        
        logger.info(f"Started transcription for stream {stream_id} in language {language_code}")
    
    def _stop_stream(self, stream_id):
        """
        Stop transcription for a stream.
        
        Args:
            stream_id: The stream ID to stop
        """
        with self._lock:
            if stream_id in self.active_streams:
                # Signal the stream to stop if possible
                stream_data = self.active_streams[stream_id]
                if hasattr(stream_data["stream"], "stop"):
                    stream_data["stream"].stop()
                del self.active_streams[stream_id]
                logger.info(f"Stopped transcription for stream {stream_id}")
    
    def _transcription_worker(self, audio_stream, stream_id, project_id, language_code, player=None):
        """
        Worker function that runs in a separate thread to process transcription.
        
        Args:
            audio_stream: The audio stream to transcribe
            stream_id: Unique identifier for this stream
            project_id: Google Cloud project ID
            language_code: Language code for transcription
            player: Optional audio player for output
        """
        try:
            logger.info(f"Transcription worker started for stream {stream_id}")
            
            # Create a request generator for this stream
            from audio_input import create_stream_request_generator
            requests_generator_func = create_stream_request_generator(
                audio_stream, 
                project_id, 
                language_code=language_code
            )
            
            # Start the streaming recognition
            responses_iterator = self.speech_client.streaming_recognize(
                requests=requests_generator_func()
            )
            
            # Process the responses
            for response in responses_iterator:
                if not self.running:
                    break
                    
                # Process the response and emit events
                self._process_response(response, stream_id, audio_stream, player)
                
        except Exception as e:
            logger.error(f"Error in transcription worker for stream {stream_id}: {e}")
            import traceback
            traceback.print_exc()
            
            # Stop timing STT in case of error
            tracker.stop_timer("stt")
            
            # Emit error event
            event_bus.publish("stt_error", {
                "stream_id": stream_id,
                "error": str(e)
            })
        finally:
            # Clean up this stream
            self._stop_stream(stream_id)
    
    def _process_response(self, response, stream_id, audio_stream, player=None):
        """
        Process a response from the streaming recognition API.
        
        Args:
            response: The API response
            stream_id: The stream ID
            audio_stream: The audio stream object
            player: Optional audio player for output
        """
        # Check if this stream has been stopped
        with self._lock:
            if stream_id not in self.active_streams:
                return
            
            echo_cancellation_active = self.active_streams[stream_id]["echo_cancellation_active"]
        
        # Handle voice activity events
        if (response.speech_event_type == 
            cloud_speech.StreamingRecognizeResponse.SpeechEventType.SPEECH_ACTIVITY_BEGIN):
            
            logger.debug("Speech activity detected")
            
            # Check if we should ignore this due to echo cancellation
            if echo_cancellation_active and config_manager.config.echo_cancellation_enabled:
                logger.debug("Ignoring speech activity due to active echo cancellation")
                return
                
            audio_stream.speech_active = True
            
            # We no longer start the timer here since we're using mic activity detection
            # to start timing earlier for more accurate measurements
            
            # Emit speech start event
            event_bus.publish("speech_start", {
                "stream_id": stream_id
            })
            
        elif (response.speech_event_type == 
              cloud_speech.StreamingRecognizeResponse.SpeechEventType.SPEECH_ACTIVITY_END):
            
            logger.debug("Speech activity ended")
            audio_stream.speech_active = False
            
            # Emit speech end event
            event_bus.publish("speech_end", {
                "stream_id": stream_id
            })
            
        # Handle transcription results
        if response.results:
            for result in response.results:
                is_final = result.is_final
                transcript = result.alternatives[0].transcript
                confidence = result.alternatives[0].confidence
                
                # Emit transcription event with explicit interim/final flag
                event_bus.publish("transcription", {
                    "stream_id": stream_id,
                    "transcript": transcript,
                    "confidence": confidence,
                    "is_final": is_final,
                    "is_interim": not is_final,  # Explicit flag for interim results
                    "timestamp": time.time(),  # Add precise timestamp
                    "language": audio_stream.language_code if hasattr(audio_stream, "language_code") else None,
                    "player": player
                })

# Legacy function maintained for backward compatibility
def transcribe_streaming_with_voice_activity(speech_client, audio_stream, project_id, language_code="en-US"):
    """
    Legacy function that transcribes audio from a stream using Google Cloud Speech-to-Text v2 API.
    Now implemented as a generator that wraps the event-based architecture.
    
    Args:
        speech_client: The Speech-to-Text v2 client
        audio_stream: The audio stream to transcribe
        project_id: The Google Cloud project ID
        language_code: The language code for transcription
        
    Yields:
        Transcription results and voice activity events
    """
    # Create a queue to receive events
    result_queue = asyncio.Queue()
    
    # Create event handlers that will put results into our queue
    def handle_speech_start(event_type, data):
        if "stream_id" in data and data["stream_id"] == stream_id:
            result_queue.put_nowait({"event": "speech_start"})
    
    def handle_speech_end(event_type, data):
        if "stream_id" in data and data["stream_id"] == stream_id:
            result_queue.put_nowait({"event": "speech_end"})
    
    def handle_transcription(event_type, data):
        if "stream_id" in data and data["stream_id"] == stream_id:
            result_queue.put_nowait({
                "event": "transcription",
                "transcript": data["transcript"],
                "confidence": data["confidence"]
            })
    
    def handle_error(event_type, data):
        if "stream_id" in data and data["stream_id"] == stream_id:
            result_queue.put_nowait({
                "event": "error",
                "error": data["error"]
            })
    
    # Generate a unique stream ID
    stream_id = id(audio_stream)
    
    # Subscribe to relevant events
    unsubscribe_funcs = [
        event_bus.subscribe("speech_start", handle_speech_start),
        event_bus.subscribe("speech_end", handle_speech_end),
        event_bus.subscribe("transcription", handle_transcription),
        event_bus.subscribe("stt_error", handle_error)
    ]
    
    try:
        # Start the transcription service if it's not already running
        # This is a simple singleton pattern for the legacy function
        if not hasattr(transcribe_streaming_with_voice_activity, "_service"):
            transcribe_streaming_with_voice_activity._service = SpeechToTextService(speech_client)
            transcribe_streaming_with_voice_activity._service.start()
            
        # Publish the audio stream to start transcription
        event_bus.publish("audio_stream_created", {
            "stream": audio_stream,
            "stream_id": stream_id,
            "project_id": project_id,
            "language_code": language_code,
            "player": audio_stream.player if hasattr(audio_stream, "player") else None  # Include player if available
        })
        
        # Wait for and yield results
        while True:
            try:
                # Use a timeout to allow for checking if we should continue
                result = asyncio.run(asyncio.wait_for(result_queue.get(), timeout=0.1))
                yield result
                result_queue.task_done()
                
                # If this was an error or we got a final transcription, we're done
                if result.get("event") in ("error", "transcription"):
                    break
            except asyncio.TimeoutError:
                # Check if the audio stream is done
                if hasattr(audio_stream, "done") and audio_stream.done:
                    break
                continue
            except Exception as e:
                logger.error(f"Error processing transcription results: {e}")
                yield {"event": "error", "error": str(e)}
                break
                
    finally:
        # Unsubscribe from events
        for unsubscribe in unsubscribe_funcs:
            unsubscribe() 