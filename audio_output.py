#!/usr/bin/env python3
"""
Audio output handling for the Live Speech Translation App.
"""

import queue
import threading
import pyaudio
import time
from modules.event_bus import event_bus

class AudioPlayer:
    """Plays audio from TTS responses with echo cancellation support."""
    
    def __init__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._play_queue = queue.Queue()
        self._playing = False
        self._stop_requested = False
        self._player_thread = threading.Thread(target=self._player_worker)
        self._player_thread.daemon = True
        self._player_thread.start()
        self._mic_stream = None  # Reference to MicrophoneStream for echo cancellation
    
    @property
    def stream_id(self):
        """Get the stream_id from the mic_stream if available."""
        if self._mic_stream:
            # Try to get the stream_id from the mic_stream
            if hasattr(self._mic_stream, "stream_id"):
                return self._mic_stream.stream_id
            # Fall back to using the id of the mic_stream object
            return id(self._mic_stream)
        return None
    
    def set_mic_stream(self, mic_stream):
        """Set the microphone stream for echo cancellation."""
        self._mic_stream = mic_stream
        # Also set the player reference in the mic stream for event passing
        if hasattr(mic_stream, 'set_player'):
            mic_stream.set_player(self)
    
    def _player_worker(self):
        """Worker thread that plays audio chunks from the queue."""
        stream = None
        try:
            while True:
                # Check if stop was requested
                if self._stop_requested:
                    # Clear the queue
                    while not self._play_queue.empty():
                        try:
                            self._play_queue.get_nowait()
                            self._play_queue.task_done()
                        except queue.Empty:
                            break
                    self._stop_requested = False
                    # Close and reopen the stream to stop current playback
                    if stream is not None:
                        stream.stop_stream()
                        stream.close()
                        stream = None
                    
                    # Notify mic stream that playback has stopped
                    if self._mic_stream:
                        self._mic_stream.register_playback_stop()
                    
                    self._playing = False
                    continue
                
                try:
                    # Use a timeout to periodically check for stop requests
                    audio_chunk = self._play_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                if audio_chunk is None:
                    break
                
                # Create stream on demand if it doesn't exist
                if stream is None:
                    stream = self._audio_interface.open(
                        format=pyaudio.paInt16,
                        channels=1,
                        rate=24000,  # TTS sample rate
                        output=True
                    )
                
                # Notify mic stream that playback is starting
                if not self._playing and self._mic_stream:
                    self._mic_stream.register_playback_start()
                    
                    # Get the stream_id for the event
                    current_stream_id = self.stream_id
                    
                    # Publish explicit playback start event
                    event_bus.publish("audio_playback_start", {
                        "stream_id": current_stream_id,
                        "pipeline_id": current_stream_id,  # Include pipeline_id as well
                        "timestamp": time.time()
                    })
                
                # Play the audio chunk
                self._playing = True
                playback_start_time = time.time()
                stream.write(audio_chunk)
                playback_end_time = time.time()
                
                # Get the stream_id for the event
                current_stream_id = self.stream_id
                
                # Publish playback completion event
                event_bus.publish("audio_chunk_played", {
                    "stream_id": current_stream_id,
                    "pipeline_id": current_stream_id,  # Include pipeline_id as well
                    "duration": playback_end_time - playback_start_time,
                    "timestamp": playback_end_time
                })
                
                # Check if queue is empty after playing this chunk
                if self._play_queue.empty():
                    # Notify mic stream that playback has stopped
                    if self._mic_stream:
                        self._mic_stream.register_playback_stop()
                    self._playing = False
                
                self._play_queue.task_done()
        finally:
            # Clean up the stream when done
            if stream is not None:
                stream.stop_stream()
                stream.close()
            
            # Ensure mic stream knows playback has stopped
            if self._playing and self._mic_stream:
                self._mic_stream.register_playback_stop()
                self._playing = False
    
    def play(self, audio_content):
        """Add audio content to the play queue."""
        self._play_queue.put(audio_content)
    
    def stop(self):
        """Stop current playback and clear the queue."""
        print("Stopping audio playback")
        self._stop_requested = True
        # Wait a short time for the worker to process the stop request
        time.sleep(0.2)
    
    def close(self):
        """Clean up resources."""
        self._play_queue.put(None)
        self._audio_interface.terminate() 