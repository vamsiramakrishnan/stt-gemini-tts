#!/usr/bin/env python3
"""
Main pipeline orchestration for the Live Speech Translation App.
"""

import asyncio
import logging
import re
import time
import threading
from typing import Any, Dict, Optional, Union, Set, List
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

# Keep only the metrics import to avoid circular dependencies
from metrics import tracker
from modules.event_bus import event_bus
from modules.config import config_manager
from modules.speech_to_text import SpeechToTextService, transcribe_streaming_with_voice_activity
from modules.text_translation import TranslationService, translate_text
from modules.text_to_speech import TextToSpeechService, stream_speech_synthesis, stream_translate_to_speech

# Set up logging
logger = logging.getLogger(__name__)

class PipelineService:
    """
    Main pipeline service that coordinates speech-to-text, translation, and text-to-speech.
    Uses event-driven architecture for loose coupling and non-blocking operation.
    Optimized for performance with asyncio and minimal locking.
    """
    
    # Singleton instance
    _instance = None
    
    @classmethod
    def get_instance(cls, speech_client=None, genai_client=None, tts_client=None):
        """Get or create the singleton instance"""
        if cls._instance is None and all([speech_client, genai_client, tts_client]):
            cls._instance = cls(speech_client, genai_client, tts_client)
        return cls._instance
    
    def __init__(self, speech_client, genai_client, tts_client):
        """
        Initialize the pipeline service.
        
        Args:
            speech_client: The Speech-to-Text client
            genai_client: The Gemini client for translation
            tts_client: The Text-to-Speech client
        """
        self.speech_client = speech_client
        self.genai_client = genai_client
        self.tts_client = tts_client
        
        # Initialize sub-services
        self.stt_service = SpeechToTextService(speech_client)
        self.translation_service = TranslationService(genai_client)
        self.tts_service = TextToSpeechService.get_instance(tts_client)
        
        self.running = False
        
        # Fine-grained locks for different parts of the pipeline state
        # This reduces contention compared to a single lock
        self._pipeline_lock = threading.RLock()
        
        # Pipeline state - segmented by pipeline ID for better parallelism
        self.active_pipelines = {}
        
        # Event handlers mapped to their unsubscribe functions for easy cleanup
        self._event_handlers = {}
        
        # Thread pool for handling computationally intensive tasks
        self._thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Define event processing queues with different priorities
        self._high_priority_events = asyncio.Queue()
        self._normal_priority_events = asyncio.Queue()
        
        # Cache for recently processed translations to avoid duplicate work
        self._translation_cache = {}
        self._cache_ttl = 300  # seconds
        
        # Cache for recent TTS operations to avoid repeated synthesis
        self._tts_cache = {}
        self._tts_cache_ttl = 60  # seconds
        
        # Performance monitoring
        self._last_pipeline_health_check = time.time()
        self._health_check_interval = 30  # seconds
        
        # Track metrics
        self.metrics = {
            "total_pipelines": 0,
            "active_pipelines": 0,
            "total_translations": 0,
            "cache_hits": 0,
            "tts_cache_hits": 0,
            "processing_latency": defaultdict(list),
            "end_to_end_latency": [],  # Speech to translation time
            "pipeline_health": 1.0     # Overall health metric (0.0-1.0)
        }
        
        # Adaptive configuration
        self._adaptive_config = {
            "min_request_interval": 0.5,  # Base rate limiting interval
            "dynamic_min_request_interval": 0.5,  # Adjusted based on system load
            "max_parallel_translations": config_manager.config.max_translation_threads,
            "echo_cancellation_delay_ms": config_manager.config.echo_cancellation_delay_ms,
            "adaptive_echo_cancellation": True,
            "speech_detection_sensitivity": 0.5  # Adjustable based on environment
        }
    
    async def _process_events(self):
        """Process events from queues according to priority"""
        while self.running:
            # Always check high priority queue first
            try:
                handler, event_type, data = self._high_priority_events.get_nowait()
                await self._run_handler(handler, event_type, data)
                self._high_priority_events.task_done()
                continue
            except asyncio.QueueEmpty:
                pass
            
            # Then check normal priority queue
            try:
                handler, event_type, data = await asyncio.wait_for(
                    self._normal_priority_events.get(), timeout=0.1
                )
                await self._run_handler(handler, event_type, data)
                self._normal_priority_events.task_done()
            except (asyncio.QueueEmpty, asyncio.TimeoutError):
                await asyncio.sleep(0.01)  # Small sleep to prevent CPU spinning
    
    async def _run_handler(self, handler, event_type, data):
        """Run an event handler and track metrics"""
        start_time = time.time()
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(event_type, data)
            else:
                # Run blocking handlers in thread pool
                await asyncio.get_event_loop().run_in_executor(
                    self._thread_pool, handler, event_type, data
                )
        except Exception as e:
            logger.error(f"Error in event handler for {event_type}: {e}", exc_info=True)
        finally:
            # Track latency
            latency = time.time() - start_time
            self.metrics["processing_latency"][event_type].append(latency)
            # Keep only recent measurements
            if len(self.metrics["processing_latency"][event_type]) > 100:
                self.metrics["processing_latency"][event_type].pop(0)
    
    def _enqueue_event(self, handler, event_type, data, high_priority=False):
        """Add an event to the appropriate queue"""
        queue = self._high_priority_events if high_priority else self._normal_priority_events
        queue.put_nowait((handler, event_type, data))
    
    def start(self):
        """Start the pipeline service and its sub-services."""
        if self.running:
            return
            
        self.running = True
        
        # Start sub-services
        self.stt_service.start()
        self.translation_service.start()
        self.tts_service.start()
        
        # Create and start the event processing task
        self._event_loop = asyncio.new_event_loop()
        self._event_thread = threading.Thread(
            target=self._run_event_loop,
            daemon=True
        )
        self._event_thread.start()
        
        # Subscribe to relevant events
        self._subscribe_to_events()
        
        # Start metrics reporting
        self._start_metrics_reporting()
        
        logger.info("Pipeline service started")
    
    def _run_event_loop(self):
        """Run the asyncio event loop in a separate thread"""
        asyncio.set_event_loop(self._event_loop)
        self._event_loop.create_task(self._process_events())
        self._event_loop.run_forever()
    
    def _subscribe_to_events(self):
        """Subscribe to all relevant events"""
        # Core pipeline control events
        self._add_event_handler("start_pipeline", self._handle_start_pipeline)
        self._add_event_handler("stop_pipeline", self._handle_stop_pipeline)
        
        # Pipeline state tracking events
        self._add_event_handler("speech_start", self._handle_speech_start)
        self._add_event_handler("speech_end", self._handle_speech_end)
        self._add_event_handler("transcription", self._handle_transcription, high_priority=True)
        self._add_event_handler("tts_active", self._handle_tts_active)
        self._add_event_handler("tts_inactive", self._handle_tts_inactive)
        
        # Add translation response handler
        self._add_event_handler("translation_response", self._handle_translation_response)
        
        # Pipeline monitoring events
        self._add_event_handler("pipeline_error", self._handle_pipeline_error)
        self._add_event_handler("component_status", self._handle_component_status)
    
    def _add_event_handler(self, event_type, handler, high_priority=False):
        """Add an event handler with priority setting"""
        wrapper = lambda evt_type, data: self._enqueue_event(handler, evt_type, data, high_priority)
        unsubscribe = event_bus.subscribe(event_type, wrapper)
        self._event_handlers[event_type] = unsubscribe
    
    def _start_metrics_reporting(self):
        """Start periodic metrics reporting"""
        def report_metrics():
            while self.running:
                with self._pipeline_lock:
                    self.metrics["active_pipelines"] = len(self.active_pipelines)
                
                # Report metrics
                for metric, value in self.metrics.items():
                    if metric == "processing_latency":
                        # Report average latencies
                        for event_type, latencies in value.items():
                            if latencies:
                                avg_latency = sum(latencies) / len(latencies)
                                # Instead of using tracker.gauge, log the metrics
                                logger.debug(f"Pipeline metric: {event_type}.latency_ms = {avg_latency * 1000:.2f}ms")
                    elif metric == "end_to_end_latency":
                        if value:
                            avg_latency = sum(value) / len(value)
                            logger.debug(f"Pipeline metric: end_to_end_latency = {avg_latency:.2f}s")
                            # Adjust adaptive configuration based on performance
                            self._adjust_adaptive_parameters(avg_latency)
                    else:
                        # Instead of using tracker.gauge, log the metrics
                        logger.debug(f"Pipeline metric: {metric} = {value}")
                
                # Clean translation cache
                self._clean_translation_cache()
                
                # Clean TTS cache
                self._clean_tts_cache()
                
                # Perform pipeline health check if interval elapsed
                current_time = time.time()
                if current_time - self._last_pipeline_health_check > self._health_check_interval:
                    self._perform_health_check()
                    self._last_pipeline_health_check = current_time
                
                time.sleep(10)  # Report every 10 seconds
        
        threading.Thread(target=report_metrics, daemon=True).start()
    
    def _adjust_adaptive_parameters(self, avg_latency):
        """Adjust adaptive parameters based on current system performance"""
        # Adjust request interval based on latency
        if avg_latency > 2.0:  # If latency is high
            self._adaptive_config["dynamic_min_request_interval"] = min(
                1.5,  # Maximum allowed interval
                self._adaptive_config["dynamic_min_request_interval"] + 0.1  # Increase interval
            )
        elif avg_latency < 0.8:  # If latency is low
            self._adaptive_config["dynamic_min_request_interval"] = max(
                0.2,  # Minimum allowed interval
                self._adaptive_config["dynamic_min_request_interval"] - 0.05  # Decrease interval
            )
    
    def _clean_translation_cache(self):
        """Clean expired entries from translation cache"""
        now = time.time()
        to_remove = [k for k, (v, timestamp) in self._translation_cache.items() 
                    if now - timestamp > self._cache_ttl]
        for key in to_remove:
            del self._translation_cache[key]
    
    def _clean_tts_cache(self):
        """Clean expired entries from TTS cache"""
        now = time.time()
        to_remove = [k for k, (_, timestamp) in self._tts_cache.items() 
                    if now - timestamp > self._tts_cache_ttl]
        for key in to_remove:
            del self._tts_cache[key]
    
    def _perform_health_check(self):
        """Perform health check on all active pipelines and components"""
        try:
            # Check component health
            stt_health = 1.0 if self.stt_service.running else 0.0
            translation_health = 1.0 if self.translation_service.running else 0.0
            tts_health = 1.0 if self.tts_service.running else 0.0
            
            # Check pipeline health
            pipeline_health = 1.0
            with self._pipeline_lock:
                # Check for dead or stalled pipelines
                current_time = time.time()
                for pipeline_id, pipeline in list(self.active_pipelines.items()):
                    # Calculate pipeline age
                    creation_time = pipeline.get("creation_time", current_time)
                    pipeline_age = current_time - creation_time
                    
                    # Check if pipeline is stalled (no activity for a long time)
                    last_activity = pipeline.get("last_activity_time", creation_time)
                    if pipeline_age > 300 and current_time - last_activity > 120:
                        logger.warning(f"Pipeline {pipeline_id} appears stalled. Stopping it.")
                        self._stop_pipeline_internal(pipeline_id)
                        del self.active_pipelines[pipeline_id]
                        pipeline_health = max(0.5, pipeline_health - 0.1)
            
            # Calculate overall health
            overall_health = (stt_health + translation_health + tts_health + pipeline_health) / 4.0
            self.metrics["pipeline_health"] = overall_health
            
            logger.info(f"Pipeline health check: {overall_health:.2f} (STT: {stt_health}, " 
                       f"Translation: {translation_health}, TTS: {tts_health}, " 
                       f"Pipeline: {pipeline_health})")
            
            # Take action if health is poor
            if overall_health < 0.5:
                logger.warning("Pipeline health is poor. Attempting recovery...")
                self._attempt_recovery()
                
        except Exception as e:
            logger.error(f"Error during health check: {e}", exc_info=True)
    
    def _attempt_recovery(self):
        """Attempt to recover from poor health conditions"""
        try:
            # Restart components if needed
            if not self.stt_service.running:
                logger.info("Restarting STT service...")
                self.stt_service.stop()
                self.stt_service = SpeechToTextService(self.speech_client)
                self.stt_service.start()
                
            if not self.translation_service.running:
                logger.info("Restarting Translation service...")
                self.translation_service.stop()
                self.translation_service = TranslationService(self.genai_client)
                self.translation_service.start()
                
            if not self.tts_service.running:
                logger.info("Restarting TTS service...")
                self.tts_service.stop()
                self.tts_service = TextToSpeechService.get_instance(self.tts_client)
                self.tts_service.start()
                
            # Clear caches to free resources
            self._translation_cache.clear()
            self._tts_cache.clear()
            
            # Reduce load by increasing request intervals
            self._adaptive_config["dynamic_min_request_interval"] = 1.0
            
            logger.info("Recovery actions completed")
            
        except Exception as e:
            logger.error(f"Error during recovery: {e}", exc_info=True)
    
    def _handle_component_status(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle component status updates
        
        Args:
            event_type: The event type
            data: Event data containing component status
        """
        if not data:
            return
            
        component = data.get("component")
        status = data.get("status")
        
        if not component or status is None:
            return
            
        logger.info(f"Component {component} status updated: {status}")
        
        # Update health metrics based on component status
        # This can be used for more fine-grained health monitoring
        pass
    
    def _handle_pipeline_error(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle pipeline error events
        
        Args:
            event_type: The event type
            data: Event data containing error information
        """
        if not data:
            return
            
        error = data.get("error")
        pipeline_id = data.get("pipeline_id")
        component = data.get("component", "unknown")
        
        logger.error(f"Pipeline error in {component} for pipeline {pipeline_id}: {error}")
        
        # Reduce health score based on severity
        severity = data.get("severity", "high")
        if severity == "high":
            self.metrics["pipeline_health"] = max(0.0, self.metrics["pipeline_health"] - 0.2)
        elif severity == "medium":
            self.metrics["pipeline_health"] = max(0.0, self.metrics["pipeline_health"] - 0.1)
        else:
            self.metrics["pipeline_health"] = max(0.0, self.metrics["pipeline_health"] - 0.05)
    
    def stop(self):
        """Stop the pipeline service and its sub-services."""
        if not self.running:
            return
            
        self.running = False
        
        # Stop sub-services in reverse order
        self.tts_service.stop()
        self.translation_service.stop()
        self.stt_service.stop()
        
        # Unsubscribe from all events
        for unsubscribe in self._event_handlers.values():
            unsubscribe()
        self._event_handlers.clear()
        
        # Stop thread pool
        self._thread_pool.shutdown(wait=False)
        
        # Stop event loop
        if hasattr(self, '_event_loop') and self._event_loop.is_running():
            self._event_loop.call_soon_threadsafe(self._event_loop.stop)
        
        # Clear active pipelines
        with self._pipeline_lock:
            for pipeline_id in list(self.active_pipelines.keys()):
                self._stop_pipeline_internal(pipeline_id)
            self.active_pipelines.clear()
        
        logger.info("Pipeline service stopped")
    
    def _stop_pipeline_internal(self, pipeline_id):
        """Internal method to stop a pipeline and clean up resources"""
        if pipeline_id not in self.active_pipelines:
            return
            
        pipeline_data = self.active_pipelines[pipeline_id]
        audio_stream = pipeline_data.get("audio_stream")
        
        # Stop the audio stream if possible
        if audio_stream:
            try:
                if hasattr(audio_stream, "stop"):
                    audio_stream.stop()
                elif hasattr(audio_stream, "close"):
                    audio_stream.close()
            except Exception as e:
                logger.error(f"Error stopping audio stream for pipeline {pipeline_id}: {e}")
        
        # Clean up any remaining resources
        if "cleanup_tasks" in pipeline_data:
            for task in pipeline_data["cleanup_tasks"]:
                try:
                    task()
                except Exception as e:
                    logger.error(f"Error in cleanup task for pipeline {pipeline_id}: {e}")
    
    async def _handle_start_pipeline(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle request to start a new pipeline.
        
        Args:
            event_type: The event type
            data: Event data containing pipeline parameters
        """
        if not self.running or not data:
            return
            
        audio_stream = data.get("audio_stream")
        if not audio_stream:
            logger.error("Cannot start pipeline: No audio stream provided")
            return
            
        # Generate a unique pipeline ID
        pipeline_id = data.get("pipeline_id", f"pipeline_{id(audio_stream)}")
        
        # Get pipeline parameters safely
        try:
            default_project_id = getattr(config_manager.config.stt_config, 'project_id', "")
        except (AttributeError, TypeError):
            default_project_id = ""
            
        project_id = data.get("project_id", default_project_id)
        source_lang = data.get("source_lang", config_manager.config.default_source_lang)
        target_lang = data.get("target_lang", config_manager.config.default_target_lang)
        player = data.get("player")
        
        # Initialize pipeline state
        with self._pipeline_lock:
            if pipeline_id in self.active_pipelines:
                logger.warning(f"Pipeline {pipeline_id} is already running")
                return
                
            self.active_pipelines[pipeline_id] = {
                "audio_stream": audio_stream,
                "project_id": project_id,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "player": player,
                "current_transcript": "",
                "is_playing_audio": False,
                "tts_active": False,
                "speech_during_tts": [],
                "last_tts_stop_time": 0,
                "last_translation_request_time": 0,
                "translations": {},  # Cache recent translations
                "cleanup_tasks": [],
                "creation_time": time.time()
            }
            
            # Update metrics
            self.metrics["total_pipelines"] += 1
        
        logger.info(f"Starting pipeline {pipeline_id}: source={source_lang}, target={target_lang}")
        
        # Publish event to start STT
        event_bus.publish("audio_stream_created", {
            "stream": audio_stream,
            "stream_id": pipeline_id,
            "project_id": project_id,
            "language_code": source_lang,
            "player": player
        })
        
        # Emit pipeline started event
        event_bus.publish("pipeline_started", {
            "pipeline_id": pipeline_id
        })
    
    async def _handle_stop_pipeline(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle request to stop a pipeline.
        
        Args:
            event_type: The event type
            data: Event data containing pipeline ID
        """
        if not data or "pipeline_id" not in data:
            return
            
        pipeline_id = data["pipeline_id"]
        
        with self._pipeline_lock:
            if pipeline_id in self.active_pipelines:
                self._stop_pipeline_internal(pipeline_id)
                
                # Remove from active pipelines
                del self.active_pipelines[pipeline_id]
                
                logger.info(f"Stopped pipeline {pipeline_id}")
                
                # Emit pipeline stopped event
                event_bus.publish("pipeline_stopped", {
                    "pipeline_id": pipeline_id
                })
    
    async def _handle_speech_start(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle speech start event with optimized echo detection.
        
        Args:
            event_type: The event type
            data: Event data containing the stream ID
        """
        if not self.running or not data or "stream_id" not in data:
            return
            
        stream_id = data["stream_id"]
        current_time = time.time()
        
        with self._pipeline_lock:
            if stream_id not in self.active_pipelines:
                return
                
            pipeline = self.active_pipelines[stream_id]
            
            # Record activity time for health monitoring
            pipeline["last_activity_time"] = current_time
            
            # Record speech start time for latency measurements
            pipeline["speech_start_time"] = current_time
            
            # Fast path for echo cancellation
            if pipeline["tts_active"]:
                # Buffer speech events during TTS
                pipeline["speech_during_tts"].append(("start", current_time))
                return
                
            # Check for echo delay threshold with adaptive value
            time_since_tts_stopped = current_time - pipeline["last_tts_stop_time"]
            echo_threshold = self._adaptive_config["echo_cancellation_delay_ms"] / 1000
            
            if time_since_tts_stopped < echo_threshold:
                logger.debug(f"Ignoring speech start due to echo delay for pipeline {stream_id}")
                return
                
            # Reset transcript for new speech segment
            pipeline["current_transcript"] = ""
            logger.debug(f"Speech started for pipeline {stream_id}")
    
    async def _handle_speech_end(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle speech end event.
        
        Args:
            event_type: The event type
            data: Event data containing the stream ID
        """
        if not self.running or not data or "stream_id" not in data:
            return
            
        stream_id = data["stream_id"]
        current_time = time.time()
        
        with self._pipeline_lock:
            if stream_id not in self.active_pipelines:
                return
                
            pipeline = self.active_pipelines[stream_id]
            
            # Fast path for echo cancellation
            if pipeline["tts_active"]:
                # Buffer speech events during TTS
                pipeline["speech_during_tts"].append(("end", current_time))
                return
                
            logger.debug(f"Speech ended for pipeline {stream_id}")
    
    async def _handle_transcription(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle transcription event with optimized processing.
        
        Args:
            event_type: The event type
            data: Event data containing the transcript
        """
        if not self.running or not data or "stream_id" not in data or "transcript" not in data:
            return
            
        stream_id = data["stream_id"]
        transcript = data["transcript"]
        current_time = time.time()
        
        # Skip processing if transcript is empty or just whitespace
        if not transcript or transcript.isspace():
            return
            
        # Quick check for pipeline existence before acquiring lock
        if stream_id not in self.active_pipelines:
            return
            
        # Get pipeline parameters for translation
        with self._pipeline_lock:
            if stream_id not in self.active_pipelines:
                return
                
            pipeline = self.active_pipelines[stream_id]
            
            # Record activity time for health monitoring
            pipeline["last_activity_time"] = current_time
            
            # Fast path for echo cancellation
            if pipeline["tts_active"]:
                logger.debug(f"Ignoring transcription during TTS for pipeline {stream_id}")
                return
                
            # Check for echo delay threshold with adaptive value
            time_since_tts_stopped = current_time - pipeline["last_tts_stop_time"]
            echo_threshold = self._adaptive_config["echo_cancellation_delay_ms"] / 1000
            
            if time_since_tts_stopped < echo_threshold:
                logger.debug(f"Ignoring transcription due to echo delay for pipeline {stream_id}")
                return
            
            # Skip if transcript hasn't changed
            if pipeline["current_transcript"] == transcript:
                return
                
            # Skip if transcript is too similar to previous (e.g. just added a period)
            prev_transcript = pipeline["current_transcript"]
            if prev_transcript and len(prev_transcript) > 5 and (
                transcript.startswith(prev_transcript[:-1]) and len(transcript) - len(prev_transcript) <= 1
            ):
                logger.debug(f"Skipping minor transcript update for pipeline {stream_id}")
                return
                
            # Update pipeline state
            pipeline["current_transcript"] = transcript
            
            # Get pipeline parameters for translation
            source_lang = pipeline["source_lang"]
            target_lang = pipeline["target_lang"]
            player = pipeline["player"]
            speech_start_time = pipeline.get("speech_start_time")
            
            # Save start time for end-to-end latency measurement if available
            if speech_start_time:
                pipeline["transcription_time"] = current_time
                transcription_latency = current_time - speech_start_time
                logger.debug(f"Transcription latency: {transcription_latency:.2f}s for pipeline {stream_id}")
            
            # Rate limiting for translation requests with adaptive interval
            time_since_last_request = current_time - pipeline.get("last_translation_request_time", 0)
            min_request_interval = self._adaptive_config["dynamic_min_request_interval"]
            
            if time_since_last_request < min_request_interval:
                logger.debug(f"Rate limiting translation request for pipeline {stream_id}")
                return
                
            pipeline["last_translation_request_time"] = current_time
        
        logger.info(f"Transcription for pipeline {stream_id}: {transcript}")
        
        # Emit pipeline transcript event for clients
        event_bus.publish("pipeline_transcript", {
            "pipeline_id": stream_id,
            "transcript": transcript,
            "source_lang": source_lang
        })
        
        # Check translation cache first
        cache_key = f"{source_lang}:{target_lang}:{transcript}"
        cached_translation = self._translation_cache.get(cache_key)
        
        if cached_translation:
            translated_text, _ = cached_translation
            self.metrics["cache_hits"] += 1
            
            # Handle cached translation directly
            await self._handle_translation_response(None, {
                "stream_id": stream_id,
                "source_text": transcript,
                "translated_text": translated_text,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "player": player,
                "from_cache": True
            })
            return
        
        # Extract language codes for translation (removing region codes if present)
        source_lang_code = source_lang.split("-")[0] if source_lang and "-" in source_lang else source_lang
        target_lang_code = target_lang.split("-")[0] if target_lang and "-" in target_lang else target_lang
        
        # Publish translation request
        event_bus.publish("translate_text", {
            "text": transcript,
            "source_lang": source_lang_code,
            "target_lang": target_lang_code,
            "stream_id": stream_id,
            "player": player
        })
        
        self.metrics["total_translations"] += 1
    
    async def _handle_translation_response(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle translation response event.
        
        Args:
            event_type: The event type
            data: Event data
        """
        if not self.running:
            return
            
        translation = data.get("translation")
        source_text = data.get("source_text")
        target_lang = data.get("target_lang")
        pipeline_id = data.get("pipeline_id")
        stream_id = data.get("stream_id")  # Make sure we get the stream_id
        
        if not translation or not pipeline_id:
            logger.warning("Missing translation or pipeline_id in translation response")
            return
            
        with self._pipeline_lock:
            if pipeline_id not in self.active_pipelines:
                logger.warning(f"Pipeline {pipeline_id} not found for translation response")
                return
                
            pipeline = self.active_pipelines[pipeline_id]
            
        # Ensure we have the stream_id from the pipeline if not provided in the event
        if not stream_id and "stream_id" in pipeline:
            stream_id = pipeline["stream_id"]
            
        # Publish translation result event
        event_bus.publish("translation_result", {
            "translation": translation,
            "source_text": source_text,
            "target_lang": target_lang,
            "pipeline_id": pipeline_id,
            "stream_id": stream_id  # Include stream_id for correlation
        })
        
        # Get the player from the pipeline
        player = pipeline.get("player")
        
        # Synthesize speech if TTS is enabled (default to enabled)
        tts_enabled = True
        try:
            # Try to access the tts_enabled property if it exists
            tts_enabled = getattr(config_manager.config, "tts_enabled", True)
        except AttributeError:
            # If it doesn't exist, default to True
            pass
            
        if tts_enabled:
            # Generate a unique callback ID for correlation
            callback_id = f"tts_{time.time()}_{pipeline_id}"
            
            # Publish synthesize speech event
            event_bus.publish("synthesize_speech", {
                "text": translation,
                "target_lang": target_lang,
                "callback_id": callback_id,
                "pipeline_id": pipeline_id,
                "stream_id": stream_id,  # Include stream_id for correlation
                "player": player
            })
    
    async def _handle_tts_active(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle TTS active event with optimized pipeline identification.
        
        Args:
            event_type: The event type
            data: Event data containing TTS information
        """
        synthesis_id = data.get("synthesis_id")
        if not synthesis_id:
            return
        
        # Find which pipeline this synthesis belongs to
        # Optimize by attaching pipeline ID to TTS events when possible
        pipeline_id = data.get("pipeline_id")
        
        with self._pipeline_lock:
            # If we have a specific pipeline ID, update just that pipeline
            if pipeline_id and pipeline_id in self.active_pipelines:
                pipeline = self.active_pipelines[pipeline_id]
                pipeline["tts_active"] = True
                pipeline["is_playing_audio"] = True
                logger.debug(f"TTS active for pipeline {pipeline_id}")
                return
                
            # Otherwise update all pipelines (legacy approach)
            for pid, pipeline in self.active_pipelines.items():
                pipeline["tts_active"] = True
                pipeline["is_playing_audio"] = True
                logger.debug(f"TTS active for pipeline {pid}")
    
    async def _handle_tts_inactive(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle TTS inactive event with optimized buffered speech processing.
        
        Args:
            event_type: The event type
            data: Event data containing TTS information
        """
        synthesis_id = data.get("synthesis_id")
        if not synthesis_id:
            return
            
        # Find which pipeline this synthesis belongs to
        # Optimize by attaching pipeline ID to TTS events when possible
        pipeline_id = data.get("pipeline_id")
        current_time = time.time()
        
        with self._pipeline_lock:
            # If we have a specific pipeline ID, update just that pipeline
            if pipeline_id and pipeline_id in self.active_pipelines:
                pipeline = self.active_pipelines[pipeline_id]
                await self._process_tts_inactive(pipeline, pipeline_id, current_time)
                return
                
            # Otherwise update all pipelines (legacy approach)
            for pid, pipeline in self.active_pipelines.items():
                await self._process_tts_inactive(pipeline, pid, current_time)
    
    async def _process_tts_inactive(self, pipeline, pipeline_id, current_time):
        """Process TTS inactive state for a specific pipeline"""
        # Mark TTS as inactive
        pipeline["tts_active"] = False
        pipeline["is_playing_audio"] = False
        pipeline["last_tts_stop_time"] = current_time
        
        # Process any buffered speech events
        if pipeline["speech_during_tts"]:
            # Intelligent processing of buffered events
            # Only keep the most recent pair of start/end events
            events = pipeline["speech_during_tts"]
            if len(events) >= 2 and events[-1][0] == "end" and any(e[0] == "start" for e in events):
                # Find the most recent start event
                start_events = [(i, e) for i, e in enumerate(events) if e[0] == "start"]
                if start_events:
                    last_start_idx, last_start = max(start_events, key=lambda x: x[1][1])
                    end_event = events[-1]
                    
                    # If there's significant time between start and end (not just noise)
                    if end_event[1] - last_start[1] > 0.5:
                        logger.debug(f"Processing meaningful buffered speech for pipeline {pipeline_id}")
                        # Reset for new speech processing
                        pipeline["current_transcript"] = ""
            
            # Clear all buffered events
            pipeline["speech_during_tts"] = []
        
        logger.debug(f"TTS inactive for pipeline {pipeline_id}")


# Legacy function for backward compatibility with simplified implementation
def process_speech_to_translation(speech_client, genai_client, tts_client, audio_stream, project_id, 
                                 source_lang="en-US", target_lang="es-US", player=None):
    """
    Legacy function that processes speech to translation pipeline.
    Now implemented to use the event-based architecture with the singleton service.
    
    Args:
        speech_client: The Speech-to-Text v2 client
        genai_client: The Gemini client for translation
        tts_client: The Text-to-Speech client
        audio_stream: The audio stream to transcribe
        project_id: The Google Cloud project ID
        source_lang: The source language code
        target_lang: The target language code
        player: The audio player for TTS output
    """
    logger.info(f"Starting speech recognition with project ID: {project_id}")
    logger.info(f"Source language: {source_lang}, Target language: {target_lang}")
    logger.info("Speak now...")
    
    # Create a pipeline through the event-based system
    pipeline_id = f"legacy_pipeline_{id(audio_stream)}"
    
    # Get the pipeline service singleton
    service = PipelineService.get_instance(speech_client, genai_client, tts_client)
    if not service.running:
        service.start()
    
    # Events for coordination
    pipeline_stopped = threading.Event()
    
    # Create event handler for pipeline stopped
    def handle_pipeline_stopped(event_type, data):
        if data.get("pipeline_id") == pipeline_id:
            pipeline_stopped.set()
    
    # Subscribe to pipeline stopped event
    unsubscribe_func = event_bus.subscribe("pipeline_stopped", handle_pipeline_stopped)
    
    try:
        # Start the pipeline
        event_bus.publish("start_pipeline", {
            "audio_stream": audio_stream,
            "pipeline_id": pipeline_id,
            "project_id": project_id,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "player": player
        })
        
        # Wait for the pipeline to be active
        while not pipeline_stopped.is_set():
            time.sleep(0.1)
            
            # Check if the audio stream is closed
            if hasattr(audio_stream, "closed") and audio_stream.closed:
                break
                
    finally:
        # Clean up by stopping the pipeline
        event_bus.publish("stop_pipeline", {
            "pipeline_id": pipeline_id
        })
        
        # Wait for pipeline to stop with timeout
        pipeline_stopped.wait(timeout=5.0)
        
        # Unsubscribe from events
        unsubscribe_func()


# Module initialization function - simplified to use singleton
def init_pipeline_service(speech_client, genai_client, tts_client):
    """
    Initialize the pipeline service.
    
    Args:
        speech_client: The Speech-to-Speech client
        genai_client: The Gemini client for translation
        tts_client: The Text-to-Speech client
        
    Returns:
        The pipeline service instance
    """
    service = PipelineService.get_instance(speech_client, genai_client, tts_client)
    if not service.running:
        service.start()
    return service 