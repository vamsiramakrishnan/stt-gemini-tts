#!/usr/bin/env python3
"""
Metrics integration for speech-to-speech translation system.
Connects the metrics tracking system to the event bus.
"""

import logging
from typing import Dict, Any, Optional
from metrics import tracker
import time

# Set up logging
logger = logging.getLogger(__name__)

class MetricsIntegration:
    """Connects the metrics tracking system to the event bus."""
    
    def __init__(self):
        """Initialize the metrics integration."""
        self.active_segment_ids = {}  # Maps stream_id to segment_id
        self.unsubscribe_funcs = []
        self.is_running = False
        self._event_bus = None
    
    @property
    def event_bus(self):
        """Lazy import of event_bus to avoid circular dependencies."""
        if self._event_bus is None:
            from modules.event_bus import event_bus
            self._event_bus = event_bus
        return self._event_bus
    
    def start(self):
        """Start tracking metrics by subscribing to events."""
        if self.is_running:
            return
            
        logger.info("Starting metrics integration")
        
        # Subscribe to relevant events
        self.unsubscribe_funcs = [
            # Speech detection and transcription events
            self.event_bus.subscribe("speech_start", self._handle_speech_start),
            self.event_bus.subscribe("transcription", self._handle_transcription),
            self.event_bus.subscribe("speech_end", self._handle_speech_end),
            
            # Microphone activity events
            self.event_bus.subscribe("mic_activity_start", self._handle_mic_activity_start),
            self.event_bus.subscribe("mic_activity_end", self._handle_mic_activity_end),
            
            # Translation events
            self.event_bus.subscribe("translate_text", self._handle_translation_start),
            self.event_bus.subscribe("translation_partial", self._handle_translation_partial),
            self.event_bus.subscribe("translation_result", self._handle_translation_complete),
            
            # TTS events
            self.event_bus.subscribe("synthesize_speech", self._handle_tts_start),
            self.event_bus.subscribe("tts_audio_chunk", self._handle_tts_audio_chunk),
            self.event_bus.subscribe("synthesis_complete", self._handle_tts_complete),
            
            # Audio processing events
            self.event_bus.subscribe("audio_processed", self._handle_audio_processed),
            
            # Audio playback events
            self.event_bus.subscribe("audio_playback_start", self._handle_audio_playback_start),
            self.event_bus.subscribe("audio_chunk_played", self._handle_audio_chunk_played)
        ]
        
        # Start auto-reporting
        tracker.start_auto_reporting()
        
        self.is_running = True
        logger.info("Metrics integration started")
    
    def stop(self):
        """Stop tracking metrics and unsubscribe from events."""
        if not self.is_running:
            return
            
        logger.info("Stopping metrics integration")
        
        # Unsubscribe from all events
        for unsubscribe in self.unsubscribe_funcs:
            unsubscribe()
        self.unsubscribe_funcs.clear()
        
        # Stop auto-reporting
        tracker.stop_auto_reporting()
        
        # Generate final report
        tracker.report()
        
        self.is_running = False
        logger.info("Metrics integration stopped")
    
    def _get_stream_id_from_data(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Extract stream_id from event data, checking for both stream_id and pipeline_id.
        
        Args:
            data: Event data
            
        Returns:
            The stream_id if found, otherwise None
        """
        # First check for stream_id
        stream_id = data.get("stream_id")
        if stream_id:
            return stream_id
            
        # Then check for pipeline_id
        pipeline_id = data.get("pipeline_id")
        if pipeline_id:
            logger.debug(f"Using pipeline_id {pipeline_id} as stream_id")
            return pipeline_id
            
        return None
    
    def _handle_speech_start(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle speech start event to track when speech is detected.
        
        Args:
            event_type: The event type
            data: Event data
        """
        stream_id = self._get_stream_id_from_data(data)
        if not stream_id:
            logger.warning(f"No stream_id or pipeline_id found in speech_start event: {data}")
            return
            
        # Track when speech is detected and generate a segment ID
        segment_id = tracker.track_speech_detected(stream_id)
        
        # Store the segment ID for future reference
        self.active_segment_ids[stream_id] = segment_id
        
        logger.debug(f"Speech detected for stream {stream_id}, created segment {segment_id}")
    
    def _handle_transcription(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle transcription event to track when transcription is received.
        
        Args:
            event_type: The event type
            data: Event data
        """
        stream_id = self._get_stream_id_from_data(data)
        if not stream_id:
            logger.warning(f"No stream_id or pipeline_id found in transcription event: {data}")
            return
        
        # Get the associated segment ID
        segment_id = self.active_segment_ids.get(stream_id)
        if not segment_id:
            logger.warning(f"No segment ID found for stream {stream_id} in transcription event")
            return
            
        # Store timestamp for logging but don't pass it to tracker methods
        timestamp = data.get("timestamp", time.time())
        
        # Determine if this is an interim or final result
        is_final = data.get("is_final", False)
        is_interim = data.get("is_interim", not is_final)
        
        logger.debug(f"Handling transcription at {timestamp:.6f} for segment {segment_id} (is_interim={is_interim}, is_final={is_final})")
        
        # Track first byte from STT (regardless of interim/final)
        # Only track this once per segment
        metrics = tracker.latency_metrics.get(segment_id)
        if metrics and not metrics.stt_first_byte_time:
            tracker.track_stt_first_byte(stream_id)
            logger.debug(f"Tracked first STT byte for segment {segment_id}")
        
        # Track interim result if this is an interim transcription
        if is_interim:
            tracker.track_stt_interim_result(stream_id)
            logger.debug(f"Tracked interim result for segment {segment_id}")
        
        # Track final result if this is a final transcription
        if is_final:
            tracker.track_stt_final_result(stream_id)
            tracker.track_stt_completed(stream_id)
            
            # If we have utterance_end_time, log the utterance-to-final latency
            if metrics and metrics.utterance_end_time:
                u2f_latency = (timestamp - metrics.utterance_end_time) * 1000
                logger.info(f"Utterance-to-final latency: {u2f_latency:.2f}ms for segment {segment_id}")
            
            logger.debug(f"Transcription completed for segment {segment_id}")
    
    def _handle_speech_end(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle speech end event.
        
        Args:
            event_type: The event type
            data: Event data
        """
        stream_id = self._get_stream_id_from_data(data)
        if not stream_id:
            logger.warning(f"No stream_id or pipeline_id found in speech_end event: {data}")
            return
        
        # Get timestamp for more accurate tracking
        timestamp = data.get("timestamp", time.time())
        
        # Track utterance end time
        tracker.track_utterance_end(stream_id)
        
        # Get the segment ID
        segment_id = self.active_segment_ids.get(stream_id)
        if not segment_id:
            logger.warning(f"No segment ID found for stream {stream_id} in speech_end event")
            return
        
        # Get metrics for this segment
        metrics = tracker.latency_metrics.get(segment_id)
        if metrics:
            # Ensure utterance_end_time is set
            if not metrics.utterance_end_time:
                metrics.utterance_end_time = timestamp
                logger.debug(f"Set utterance_end_time for segment {segment_id} to {timestamp:.6f}")
        
        # Mark STT as completed if not already done
        segment_id = tracker.track_stt_completed(stream_id)
        
        if segment_id:
            logger.debug(f"Speech ended for segment {segment_id} at {timestamp:.6f}")
    
    def _handle_translation_start(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle translation start event.
        
        Args:
            event_type: The event type
            data: Event data
        """
        stream_id = self._get_stream_id_from_data(data)
        if not stream_id:
            logger.warning(f"No stream_id or pipeline_id found in translation_start event: {data}")
            return
            
        if stream_id not in self.active_segment_ids:
            logger.warning(f"No segment ID found for stream {stream_id} in translation_start event")
            return
            
        segment_id = self.active_segment_ids[stream_id]
        
        # Track translation start
        tracker.track_translation_start(segment_id)
        
        logger.debug(f"Translation started for segment {segment_id}")
    
    def _handle_translation_partial(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle partial translation event to track tokens and meaningful chunks.
        
        Args:
            event_type: The event type
            data: Event data
        """
        stream_id = self._get_stream_id_from_data(data)
        if not stream_id:
            logger.warning(f"No stream_id or pipeline_id found in translation_partial event: {data}")
            return
            
        if stream_id not in self.active_segment_ids:
            logger.warning(f"No segment ID found for stream {stream_id} in translation_partial event")
            return
            
        segment_id = self.active_segment_ids[stream_id]
        
        partial_text = data.get("partial_text", "")
        is_meaningful = data.get("is_meaningful", len(partial_text.split()) >= 3)  # Consider 3+ words meaningful
        
        # Store timestamp for logging but don't pass it to tracker methods
        timestamp = data.get("timestamp", time.time())
        
        # Get token count from data
        token_count = data.get("token_count", 1)
        
        logger.debug(f"Handling translation partial at {timestamp:.6f} for segment {segment_id}, token_count={token_count}")
        
        # Get metrics for this segment
        metrics = tracker.latency_metrics.get(segment_id)
        
        # Track first token if this is the first token
        if metrics and not metrics.translation_first_token_time:
            tracker.track_translation_first_token(segment_id)
            logger.debug(f"Tracked first translation token for segment {segment_id}")
        
        # Track token generation
        tracker.track_translation_token(segment_id)
        
        # Update token count and last token time for rate calculation
        if metrics:
            # Initialize token count if not already set
            if not hasattr(metrics, 'translation_token_count'):
                metrics.translation_token_count = 0
            
            # Update token count
            metrics.translation_token_count = token_count
            
            # Update last token time
            metrics.translation_last_token_time = timestamp
            
            # Calculate and store token rate if we have both first token time and last token time
            if hasattr(metrics, 'translation_first_token_time') and metrics.translation_first_token_time and token_count > 1:
                time_diff = timestamp - metrics.translation_first_token_time
                if time_diff > 0:
                    token_rate = (token_count - 1) / time_diff  # tokens per second
                    # Store token rate in metrics
                    metrics.translation_token_rate = token_rate
                    logger.debug(f"Token rate for segment {segment_id}: {token_rate:.2f} tokens/sec")
        
        # Track meaningful chunks explicitly
        if is_meaningful and metrics and not metrics.translation_first_meaningful_time:
            tracker.track_translation_meaningful_chunk(segment_id)
            logger.debug(f"Tracked meaningful translation chunk for segment {segment_id}")
        
        logger.debug(f"Translation partial for segment {segment_id}: {partial_text[:30]}...")
    
    def _handle_translation_complete(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle translation complete event.
        
        Args:
            event_type: The event type
            data: Event data
        """
        stream_id = self._get_stream_id_from_data(data)
        if not stream_id:
            logger.warning(f"No stream_id or pipeline_id found in translation_complete event: {data}")
            return
            
        if stream_id not in self.active_segment_ids:
            logger.warning(f"No segment ID found for stream {stream_id} in translation_complete event")
            return
            
        segment_id = self.active_segment_ids[stream_id]
        
        # Track translation completion
        tracker.track_translation_completed(segment_id)
        
        logger.debug(f"Translation completed for segment {segment_id}")
    
    def _handle_tts_start(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle TTS start event.
        
        Args:
            event_type: The event type
            data: Event data
        """
        # TTS events may not have the stream_id, but should have callback_id
        # which correlates to a previous translation event
        stream_id = self._get_stream_id_from_data(data)
        callback_id = data.get("callback_id")
        
        # Try to find segment_id using stream_id
        segment_id = None
        if stream_id and stream_id in self.active_segment_ids:
            segment_id = self.active_segment_ids[stream_id]
        
        # If we have a segment_id, track TTS start
        if segment_id:
            tracker.track_tts_start(segment_id)
            logger.debug(f"TTS started for segment {segment_id}")
        else:
            logger.warning(f"No segment ID found for stream {stream_id} in tts_start event")
    
    def _handle_tts_audio_chunk(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle TTS audio chunk event to track first audio byte and playback.
        
        Args:
            event_type: The event type
            data: Event data
        """
        stream_id = self._get_stream_id_from_data(data)
        callback_id = data.get("callback_id")
        metrics_only = data.get("metrics_only", False)
        playback_time = data.get("playback_time")
        generation_time = data.get("generation_time")
        is_first_chunk = data.get("is_first_chunk", True)  # Default to True for backward compatibility
        
        # Prioritize playback time over generation time
        timestamp = data.get("playback_time") or data.get("generation_time") or time.time()
        
        # Try to find segment_id using stream_id
        segment_id = None
        if stream_id and stream_id in self.active_segment_ids:
            segment_id = self.active_segment_ids[stream_id]
            
            # If we have a segment_id and this is the first chunk, track TTS first audio byte
            if segment_id and is_first_chunk:
                # Track the first audio byte
                tracker.track_tts_first_audio_byte(segment_id)
                
                # Update the timestamp manually for more accuracy
                metrics = tracker.latency_metrics.get(segment_id)
                if metrics:
                    metrics.tts_first_audio_byte_time = timestamp
                    
                    # Calculate and log end-to-end latency
                    if metrics.speech_detected_time:
                        e2e_latency = (timestamp - metrics.speech_detected_time) * 1000
                        logger.info(f"End-to-end latency: {e2e_latency:.2f}ms for segment {segment_id}")
                    
                    # Calculate and log utterance-to-audio latency if utterance_end_time is available
                    if metrics.utterance_end_time:
                        u2a_latency = (timestamp - metrics.utterance_end_time) * 1000
                        logger.info(f"Utterance-to-audio latency: {u2a_latency:.2f}ms for segment {segment_id}")
                
                logger.debug(f"TTS first audio byte for segment {segment_id} at time {timestamp:.6f}")
        else:
            # If we couldn't find the segment_id using stream_id, log a warning
            logger.warning(f"Could not find segment_id for stream_id={stream_id} in active_segment_ids={self.active_segment_ids}")
            
            # Try to recover by looking for a recent segment
            recent_segments = tracker.get_recent_segments(5)
            if recent_segments and is_first_chunk:
                # Use the most recent segment as a fallback
                fallback_segment_id = recent_segments[0].get("segment_id")
                logger.info(f"Using fallback segment_id {fallback_segment_id} for TTS metrics")
                
                # Track TTS first audio byte with the fallback segment
                tracker.track_tts_first_audio_byte(fallback_segment_id)
                
                # Update the timestamp manually if possible
                fallback_metrics = tracker.latency_metrics.get(fallback_segment_id)
                if fallback_metrics:
                    fallback_metrics.tts_first_audio_byte_time = timestamp
    
    def _handle_tts_complete(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle TTS complete event.
        
        Args:
            event_type: The event type
            data: Event data
        """
        stream_id = self._get_stream_id_from_data(data)
        callback_id = data.get("callback_id")
        
        # Try to find segment_id using stream_id
        segment_id = None
        if stream_id and stream_id in self.active_segment_ids:
            segment_id = self.active_segment_ids[stream_id]
        
        # If we have a segment_id, track TTS completion
        if segment_id:
            tracker.track_tts_completed(segment_id)
            logger.debug(f"TTS completed for segment {segment_id}")
            
            # Segment is now complete, we can clean it up
            tracker.complete_speech_segment(stream_id)
            if stream_id in self.active_segment_ids:
                del self.active_segment_ids[stream_id]
        else:
            logger.warning(f"No segment ID found for stream {stream_id} in tts_complete event")
    
    def _handle_audio_processed(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle audio processed event.
        
        Args:
            event_type: The event type
            data: Event data
        """
        stream_id = self._get_stream_id_from_data(data)
        bytes_count = data.get("bytes_count", 0)
        
        if stream_id and bytes_count > 0:
            tracker.track_audio_processed(stream_id, bytes_count)

    def _handle_mic_activity_start(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle microphone activity start event to track when actual audio input begins.
        This provides a more accurate starting point for timing than waiting for speech detection.
        
        Args:
            event_type: The event type
            data: Event data
        """
        stream_id = self._get_stream_id_from_data(data)
        if not stream_id:
            logger.warning(f"No stream_id or pipeline_id found in mic_activity_start event: {data}")
            return
            
        # Check if we already have a segment for this stream
        if stream_id in self.active_segment_ids:
            # We already have a segment, so we don't need to create a new one
            logger.debug(f"Mic activity detected for existing segment {self.active_segment_ids[stream_id]}")
            return
            
        # Create a new segment or use existing one
        segment_id = tracker.track_mic_activity(stream_id)
        
        # Store the segment ID with an explicit timestamp
        self.active_segment_ids[stream_id] = segment_id
        
        # Log the exact timestamp for debugging
        logger.debug(f"Mic activity detected at {time.time():.6f} for segment {segment_id}")
    
    def _handle_mic_activity_end(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle microphone activity end event.
        
        Args:
            event_type: The event type
            data: Event data
        """
        stream_id = self._get_stream_id_from_data(data)
        if not stream_id:
            logger.warning(f"No stream_id or pipeline_id found in mic_activity_end event: {data}")
            return
            
        logger.debug(f"Mic activity ended for stream {stream_id}")
        
        # We don't complete the segment here, as we still need to wait for speech recognition
        # to complete. The segment will be completed when we receive the speech_end event.

    def _handle_audio_playback_start(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle audio playback start event to track when audio starts playing.
        
        Args:
            event_type: The event type
            data: Event data
        """
        stream_id = self._get_stream_id_from_data(data)
        timestamp = data.get("timestamp", time.time())
        
        if not stream_id:
            logger.warning(f"No stream_id or pipeline_id found in audio_playback_start event: {data}")
            return
            
        # Try to find segment_id using stream_id
        segment_id = self.active_segment_ids.get(stream_id)
        if not segment_id:
            # Try to recover by looking for a recent segment
            recent_segments = tracker.get_recent_segments(5)
            if recent_segments:
                # Use the most recent segment as a fallback
                segment_id = recent_segments[0].get("segment_id")
                logger.info(f"Using fallback segment_id {segment_id} for audio playback metrics")
        
        if segment_id:
            # If we have a segment_id, track TTS first audio byte if not already tracked
            metrics = tracker.latency_metrics.get(segment_id)
            if metrics and not metrics.tts_first_audio_byte_time:
                # Update with precise timestamp
                metrics.tts_first_audio_byte_time = timestamp
                
                # Calculate and log end-to-end latency
                if metrics.speech_detected_time:
                    e2e_latency = (timestamp - metrics.speech_detected_time) * 1000
                    logger.info(f"End-to-end latency: {e2e_latency:.2f}ms for segment {segment_id}")
                
                logger.debug(f"Audio playback started for segment {segment_id} at {timestamp:.6f}")
        else:
            logger.warning(f"No segment ID found for stream {stream_id} in audio_playback_start event")
    
    def _handle_audio_chunk_played(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle audio chunk played event to track when audio finishes playing.
        
        Args:
            event_type: The event type
            data: Event data
        """
        stream_id = self._get_stream_id_from_data(data)
        timestamp = data.get("timestamp", time.time())
        
        if not stream_id:
            logger.warning(f"No stream_id or pipeline_id found in audio_chunk_played event: {data}")
            return
            
        # Try to find segment_id using stream_id
        segment_id = self.active_segment_ids.get(stream_id)
        if not segment_id:
            logger.warning(f"No segment ID found for stream {stream_id} in audio_chunk_played event")
            return
            
        logger.debug(f"Audio chunk played for segment {segment_id} at {timestamp:.6f}")

# Create a singleton instance
metrics_integration = MetricsIntegration() 