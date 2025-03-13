#!/usr/bin/env python3
"""
Audio input handling for the Live Speech Translation App.
"""

import queue
import pyaudio
import os
import numpy as np
import time
import logging
from config import RATE, CHUNK, STREAMING_LIMIT, get_current_time
from google.cloud.speech_v2.types import cloud_speech
# Remove metrics_integration import to avoid circular imports
from modules.event_bus import event_bus

# Set up logging
logger = logging.getLogger(__name__)

class MicrophoneStream:
    """Opens a recording stream as a generator yielding the audio chunks with echo cancellation."""

    def __init__(self, rate=RATE, chunk=CHUNK):
        self._rate = rate
        self._chunk = chunk
        self._buff = queue.Queue()
        self.closed = True
        # Tracking variables for stream management
        self.start_time = get_current_time()
        self.restart_counter = 0
        self.audio_input = []
        self.last_audio_input = []
        self.result_end_time = 0
        self.is_final_end_time = 0
        self.final_request_end_time = 0
        self.bridging_offset = 0
        self.last_transcript_was_final = False
        self.new_stream = True
        # Voice activity tracking
        self.speech_active = False
        
        # Echo cancellation variables
        self.is_playing_audio = False
        self.playback_buffer = []
        self.playback_buffer_lock = None  # Will be initialized in __enter__
        self.echo_suppression_active = True
        self.energy_threshold = 500  # Threshold for detecting audio energy
        self.last_playback_time = 0
        self.echo_suppression_duration = 0.5  # Duration in seconds to suppress after playback
        
        # Mic activity detection variables
        self.mic_active = False
        self.mic_activity_threshold = 300  # Threshold for detecting microphone activity
        self.mic_activity_cooldown = 1.0  # Cooldown period in seconds to avoid rapid toggling
        self.last_mic_activity_time = 0
        
        # Store player reference for use with event bus
        self.player = None
        
        # Store the stream ID for consistent reference
        self._stream_id = id(self)

    @property
    def stream_id(self):
        """Get the unique ID for this stream."""
        return self._stream_id

    def __enter__(self):
        import threading
        self.playback_buffer_lock = threading.Lock()
        
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer,
        )

        self.closed = False
        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """
        Continuously collect data from the audio stream, into the buffer.
        Applies echo cancellation if audio is playing.
        """
        # Calculate audio energy for mic activity detection
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        mean_squared = max(1e-10, np.mean(audio_data.astype(np.float32)**2))
        energy = np.sqrt(mean_squared)

        # Emit audio processed event with more metadata
        event_bus.publish("audio_processed", {
            "stream_id": self.stream_id,
            "pipeline_id": self.stream_id,  # Include pipeline_id as well
            "bytes_count": len(in_data),
            "timestamp": time.time(),  # Add precise timestamp
            "energy": energy  # Include energy level for debugging
        })
        
        # Calculate audio energy for mic activity detection - fix to avoid invalid sqrt values
        # Use max(1e-10, value) to ensure we never take sqrt of zero or negative values
        mean_squared = max(1e-10, np.mean(audio_data.astype(np.float32)**2))
        energy = np.sqrt(mean_squared)
        current_time = time.time()
        
        # Check for microphone activity (only if not in echo cancellation mode)
        if not self.is_playing_audio and (current_time - self.last_playback_time > self.echo_suppression_duration):
            # If energy is above threshold and we're not in cooldown period, mark as active
            if energy > self.mic_activity_threshold and (current_time - self.last_mic_activity_time > self.mic_activity_cooldown):
                if not self.mic_active:
                    self.mic_active = True
                    self.last_mic_activity_time = current_time
                    # Emit mic activity start event
                    event_bus.publish("mic_activity_start", {
                        "stream_id": self.stream_id,
                        "pipeline_id": self.stream_id,  # Include pipeline_id as well
                        "energy": energy
                    })
                    logger.debug(f"Microphone activity detected with energy: {energy}")
            # If energy is below threshold and we're active, mark as inactive after cooldown
            elif energy < self.mic_activity_threshold * 0.7 and self.mic_active and (current_time - self.last_mic_activity_time > self.mic_activity_cooldown):
                self.mic_active = False
                self.last_mic_activity_time = current_time
                # Emit mic activity end event
                event_bus.publish("mic_activity_end", {
                    "stream_id": self.stream_id,
                    "pipeline_id": self.stream_id,  # Include pipeline_id as well
                    "energy": energy
                })
                logger.debug(f"Microphone activity ended with energy: {energy}")
        
        # Apply echo cancellation if needed
        if self.echo_suppression_active:
            # Check if we're currently playing audio or recently played audio
            if self.is_playing_audio or (time.time() - self.last_playback_time < self.echo_suppression_duration):
                # Calculate audio energy - fix to avoid invalid sqrt values
                mean_squared = max(1e-10, np.mean(audio_data.astype(np.float32)**2))
                energy = np.sqrt(mean_squared)
                
                # If energy is below threshold or we're definitely playing back, apply suppression
                if energy < self.energy_threshold or self.is_playing_audio:
                    # Apply strong suppression during playback
                    if self.is_playing_audio:
                        # Almost completely suppress input during playback
                        audio_data = np.zeros_like(audio_data)
                    else:
                        # Apply gentler suppression during echo tail
                        # Calculate a fade-in factor based on time since playback ended
                        time_since_playback = time.time() - self.last_playback_time
                        fade_in_factor = min(1.0, time_since_playback / self.echo_suppression_duration)
                        audio_data = (audio_data * fade_in_factor).astype(np.int16)
        
        # Put the processed audio data into the buffer
        self._buff.put(audio_data.tobytes())
        return None, pyaudio.paContinue

    def register_playback_start(self):
        """Register that audio playback has started."""
        self.is_playing_audio = True
        
    def register_playback_stop(self):
        """Register that audio playback has stopped."""
        self.is_playing_audio = False
        self.last_playback_time = time.time()

    def set_echo_suppression(self, active):
        """Enable or disable echo suppression."""
        self.echo_suppression_active = active
        
    def set_player(self, player):
        """Set the player reference for use with event bus."""
        self.player = player

    def generator(self):
        """Generator yielding chunks of audio data with improved handling and reduced latency."""
        while not self.closed:
            # Handle stream restarts by reusing previous audio
            if self.new_stream and self.last_audio_input:
                chunk_time = STREAMING_LIMIT / len(self.last_audio_input)
                
                if chunk_time != 0:
                    if self.bridging_offset < 0:
                        self.bridging_offset = 0
                    
                    if self.bridging_offset > self.final_request_end_time:
                        self.bridging_offset = self.final_request_end_time
                    
                    chunks_from_ms = round(
                        (self.final_request_end_time - self.bridging_offset) / chunk_time
                    )
                    
                    self.bridging_offset = round(
                        (len(self.last_audio_input) - chunks_from_ms) * chunk_time
                    )
                    
                    # Initialize data before using it
                    data = []
                    for i in range(chunks_from_ms, len(self.last_audio_input)):
                        data.append(self.last_audio_input[i])
                
                self.new_stream = False
            else:
                # Use a shorter timeout to reduce latency
                try:
                    chunk = self._buff.get(timeout=0.1)
                except queue.Empty:
                    continue
                    
                if chunk is None:
                    return
                data = [chunk]

                # Use a non-blocking approach with a limit to avoid excessive buffering
                # This prevents accumulating too much audio and causing latency
                max_chunks = 5  # Limit chunks to process at once
                chunks_processed = 1
                
                while chunks_processed < max_chunks:
                    try:
                        chunk = self._buff.get(block=False)
                        if chunk is None:
                            return
                        data.append(chunk)
                        chunks_processed += 1
                    except queue.Empty:
                        break

            yield b"".join(data)

def create_stream_request_generator(stream, project_id, language_code="en-US", model="latest_long"):
    """Creates a request generator for the Speech-to-Text v2 API.
    
    Args:
        stream: The audio stream to transcribe
        project_id: The Google Cloud project ID
        language_code: The language code for transcription
        model: The model to use for transcription
        
    Returns:
        A function that generates StreamingRecognizeRequest objects
    """
    # Create the recognition config with explicit encoding
    recognition_config = cloud_speech.RecognitionConfig(
        # Use explicit decoding config instead of auto
        explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
            encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            audio_channel_count=1
        ),
        language_codes=[language_code],
        model=model,
    )
    
    # Enable voice activity events but disable interim results
    streaming_features = cloud_speech.StreamingRecognitionFeatures(
        enable_voice_activity_events=True,
        interim_results=False  # Disable interim results for faster processing
    )
    
    streaming_config = cloud_speech.StreamingRecognitionConfig(
        config=recognition_config, 
        streaming_features=streaming_features
    )
    
    # Create the initial config request with explicit recognizer path
    recognizer_path = f"projects/{project_id}/locations/global/recognizers/_"
    print(f"Using recognizer path: {recognizer_path}")
    
    config_request = cloud_speech.StreamingRecognizeRequest(
        recognizer=recognizer_path,
        streaming_config=streaming_config,
    )
    
    # This follows the exact pattern from the reference code
    def requests_generator():
        # First yield the config request
        print("Sending config request...")
        yield config_request
        
        # Then yield audio content requests from the stream
        print("Starting to send audio data...")
        for content in stream.generator():
            # Print a dot to show we're getting audio data
            print(".", end="", flush=True)
            yield cloud_speech.StreamingRecognizeRequest(audio=content)
    
    # Return the generator function
    return requests_generator 