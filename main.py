#!/usr/bin/env python3
"""
Main entry point for the Live Speech Translation App.
"""

import re
import time
import threading
import logging
import sys
import signal
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech

from config import RATE, CHUNK, STREAMING_LIMIT, setup_clients, get_current_time, get_project_id
from audio_input import MicrophoneStream
from audio_output import AudioPlayer
# Remove direct metrics import and only use metrics_integration
from metrics_integration import metrics_integration

# Import our new modular components
from modules import (
    event_bus, 
    config_manager, 
    init_pipeline_service,
    process_speech_to_translation
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Handle graceful shutdown
shutdown_event = threading.Event()

def signal_handler(sig, frame):
    """Handle shutdown signals gracefully."""
    logger.info("Shutdown signal received, cleaning up...")
    shutdown_event.set()
    
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def listen_and_translate():
    """
    Main function to listen, transcribe, translate, and speak.
    Now using the event-driven architecture for better modularity and performance.
    """
    # Set up clients in parallel to speed up initialization
    clients_ready = threading.Event()
    clients = [None, None, None]  # [speech_client, genai_client, tts_client]
    
    def init_clients():
        clients[0], clients[1], clients[2] = setup_clients()
        clients_ready.set()
    
    # Start client initialization in background
    client_thread = threading.Thread(target=init_clients)
    client_thread.daemon = True
    client_thread.start()
    
    logger.info("Initializing clients...")
    
    # Get language selection from user while clients are initializing
    print("Available target languages:")
    print("1. Spanish (es-US)")
    print("2. French (fr-FR)")
    print("3. German (de-DE)")
    print("4. Italian (it-IT)")
    print("5. Japanese (ja-JP)")
    print("6. Hindi (hi-IN)")
    choice = input("Select target language (1-6, default is 1): ")
    
    target_lang_map = {
        "1": "es-US",
        "2": "fr-FR",
        "3": "de-DE",
        "4": "it-IT",
        "5": "ja-JP",
        "6": "hi-IN"
    }
    
    target_lang = target_lang_map.get(choice, "es-US")
    
    # Update configuration with user preferences
    config_manager.update_config(default_target_lang=target_lang)
    
    # Get project ID
    project_id = get_project_id()
    # Update the configuration with the project ID using dot notation syntax
    config_manager.update_config(**{"stt_config.project_id": project_id})
    logger.info(f"Using project ID: {project_id}")
    
    # Wait for clients to be ready
    clients_ready.wait()
    speech_client, genai_client, tts_client = clients
    
    # Initialize audio player
    player = AudioPlayer()
    
    # Test the audio player to make sure it's working
    logger.info("Testing audio player...")
    try:
        # Create a simple audio test (a sine wave)
        import numpy as np
        sample_rate = 24000
        duration = 0.5  # half-second beep
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        test_audio = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16).tobytes()
        
        # Play the test audio
        player.play(test_audio)
        logger.info("Test audio queued for playback, if you hear a beep, audio output is working")
        time.sleep(1)  # Give time for the audio to play
    except Exception as e:
        logger.error(f"Audio player test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test direct TTS functionality with Hindi
    logger.info("Testing Hindi TTS directly...")
    try:
        from modules.text_to_speech import stream_speech_synthesis
        test_hindi_text = "नमस्ते, यह एक परीक्षण है"  # "Hello, this is a test" in Hindi
        stream_speech_synthesis(tts_client, test_hindi_text, "hi-IN", player)
        logger.info("Hindi TTS test initiated, if you hear Hindi speech, TTS is working")
        time.sleep(3)  # Give time for the synthesis and playback
    except Exception as e:
        logger.error(f"Hindi TTS test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Ask user if they want to enable echo cancellation
    echo_choice = input("Enable echo cancellation? (y/n, default: y): ").lower()
    echo_cancellation = echo_choice != 'n'
    
    # Update configuration with echo cancellation preference
    config_manager.update_config(echo_cancellation_enabled=echo_cancellation)
    
    logger.info(f"Translating to {target_lang}. Speak in English. Say 'exit' to quit.")
    
    # Initialize variables before we enter the try block to fix linter errors
    pipeline_service = None
    unsubscribe_funcs = []
    
    try:
        # Initialize the pipeline service
        pipeline_service = init_pipeline_service(speech_client, genai_client, tts_client)
        
        # Start the metrics integration
        metrics_integration.start()
        
        # Set up event handlers for pipeline events
        def handle_transcript(event_type, data):
            """Handle transcript events from the pipeline."""
            if "transcript" in data:
                logger.info(f"Transcript: {data['transcript']}")
                
                # Check for exit command
                if re.search(r"\b(exit|quit)\b", data["transcript"], re.I):
                    logger.info("Exit command detected")
                    shutdown_event.set()
        
        def handle_translation(event_type, data):
            """Handle translation events from the pipeline."""
            if "translation" in data:
                logger.info(f"Translation: {data['translation']}")
                
        def handle_tts_audio_chunk(event_type, data):
            """Handle TTS audio chunk events for debugging."""
            audio_content = data.get("audio_content")
            if audio_content:
                logger.debug(f"Received TTS audio chunk: {len(audio_content)} bytes")
        
        # Subscribe to relevant events
        unsubscribe_funcs = [
            event_bus.subscribe("pipeline_transcript", handle_transcript),
            event_bus.subscribe("translation_result", handle_translation),
            event_bus.subscribe("tts_audio_chunk", handle_tts_audio_chunk)
        ]
        
        # Pre-initialize the pipeline components to reduce latency
        logger.info("Pre-initializing pipeline components to reduce latency...")
        
        # Pre-warm the STT service with a dummy request
        event_bus.publish("prewarm_stt", {
            "project_id": project_id,
            "language_code": "en-US"
        })
        
        # Pre-warm the translation service with a dummy request
        event_bus.publish("prewarm_translation", {
            "source_lang": "en",
            "target_lang": target_lang.split("-")[0]
        })
        
        # Pre-warm the TTS service with a dummy request
        event_bus.publish("prewarm_tts", {
            "target_lang": target_lang
        })
        
        # Give some time for the pre-warming to complete
        time.sleep(1)
        logger.info("Pipeline components pre-initialized")
        
        logger.info("Initializing microphone stream...")
        with MicrophoneStream(RATE, CHUNK) as stream:
            logger.info("Microphone stream initialized.")
            
            # Connect the audio player to the microphone stream for echo cancellation
            player.set_mic_stream(stream)
            
            # Log player information
            logger.debug(f"Audio player initialized: {player}")
            logger.debug(f"MicrophoneStream player reference: {getattr(stream, 'player', None)}")
            
            # Enable or disable echo suppression based on user preference
            stream.set_echo_suppression(echo_cancellation)
            if echo_cancellation:
                logger.info("Echo cancellation enabled - system will ignore its own audio output")
            else:
                logger.info("Echo cancellation disabled")
            
            # Generate a unique pipeline ID
            pipeline_id = f"main_pipeline_{id(stream)}"
            
            # Start the pipeline through the event system
            event_bus.publish("start_pipeline", {
                "audio_stream": stream,
                "pipeline_id": pipeline_id,
                "project_id": project_id,
                "source_lang": "en-US",
                "target_lang": target_lang,
                "player": player
            })
            
            # Keep the main thread running until shutdown is requested
            logger.info("Pipeline started, waiting for speech input...")
            while not shutdown_event.is_set() and not stream.closed:
                # Check if the stream has been marked as closed (due to "exit" command)
                if hasattr(stream, "closed") and stream.closed:
                    logger.info("Stream closed, stopping pipeline")
                    break
                    
                # Sleep briefly to avoid high CPU usage
                time.sleep(0.1)
            
            # Stop the pipeline gracefully
            logger.info("Stopping pipeline...")
            event_bus.publish("stop_pipeline", {
                "pipeline_id": pipeline_id
            })
            
    except KeyboardInterrupt:
        logger.info("\nExiting on user request (Ctrl+C)")
    except Exception as e:
        logger.error(f"Error in listen_and_translate: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Unsubscribe from events to prevent memory leaks
        for unsubscribe in unsubscribe_funcs:
            unsubscribe()
        
        # Stop the metrics integration and generate final report
        metrics_integration.stop()
        
        # Stop the pipeline service if it exists
        if pipeline_service is not None:
            pipeline_service.stop()

def legacy_listen_and_translate():
    """
    Legacy function using the old direct function calls.
    Kept for backward compatibility.
    """
    # Set up clients in parallel to speed up initialization
    clients_ready = threading.Event()
    clients = [None, None, None]  # [speech_client, genai_client, tts_client]
    
    def init_clients():
        clients[0], clients[1], clients[2] = setup_clients()
        clients_ready.set()
    
    # Start client initialization in background
    client_thread = threading.Thread(target=init_clients)
    client_thread.daemon = True
    client_thread.start()
    
    print("Initializing clients...")
    
    # Get language selection from user while clients are initializing
    print("Available target languages:")
    print("1. Spanish (es-US)")
    print("2. French (fr-FR)")
    print("3. German (de-DE)")
    print("4. Italian (it-IT)")
    print("5. Japanese (ja-JP)")
    print("6. Hindi (hi-IN)")
    choice = input("Select target language (1-6, default is 1): ")
    
    target_lang_map = {
        "1": "es-US",
        "2": "fr-FR",
        "3": "de-DE",
        "4": "it-IT",
        "5": "ja-JP",
        "6": "hi-IN"
    }
    
    target_lang = target_lang_map.get(choice, "es-US")
    
    # Get project ID
    project_id = get_project_id()
    print(f"Using project ID: {project_id}")
    
    # Wait for clients to be ready
    clients_ready.wait()
    speech_client, genai_client, tts_client = clients
    
    # Initialize audio player
    player = AudioPlayer()
    
    # Test the audio player to make sure it's working
    logger.info("Testing audio player...")
    try:
        # Create a simple audio test (a sine wave)
        import numpy as np
        sample_rate = 24000
        duration = 0.5  # half-second beep
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        test_audio = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16).tobytes()
        
        # Play the test audio
        player.play(test_audio)
        logger.info("Test audio queued for playback, if you hear a beep, audio output is working")
        time.sleep(1)  # Give time for the audio to play
    except Exception as e:
        logger.error(f"Audio player test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test direct TTS functionality with Hindi
    logger.info("Testing Hindi TTS directly...")
    try:
        from modules.text_to_speech import stream_speech_synthesis
        test_hindi_text = "नमस्ते, यह एक परीक्षण है"  # "Hello, this is a test" in Hindi
        stream_speech_synthesis(tts_client, test_hindi_text, "hi-IN", player)
        logger.info("Hindi TTS test initiated, if you hear Hindi speech, TTS is working")
        time.sleep(3)  # Give time for the synthesis and playback
    except Exception as e:
        logger.error(f"Hindi TTS test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Ask user if they want to enable echo cancellation
    echo_cancellation = input("Enable echo cancellation? (y/n, default: y): ").lower() != 'n'
    
    print(f"Translating to {target_lang}. Speak in English. Say 'exit' to quit.")
    print("Initializing microphone stream...")
    
    try:
        with MicrophoneStream(RATE, CHUNK) as stream:
            print("Microphone stream initialized.")
            
            # Connect the audio player to the microphone stream for echo cancellation
            player.set_mic_stream(stream)
            
            # Enable or disable echo suppression based on user preference
            stream.set_echo_suppression(echo_cancellation)
            if echo_cancellation:
                print("Echo cancellation enabled - system will ignore its own audio output")
            else:
                print("Echo cancellation disabled")
            
            # Use the process_speech_to_translation function to handle the entire pipeline
            process_speech_to_translation(
                speech_client,
                genai_client,
                tts_client,
                stream,
                project_id,
                source_lang="en-US",
                target_lang=target_lang,
                player=player
            )
    except KeyboardInterrupt:
        print("\nExiting on user request (Ctrl+C)")
    except Exception as e:
        print(f"Error in listen_and_translate: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Choose which implementation to use
    use_event_driven = True  # Set to False to use the legacy approach
    
    if use_event_driven:
        listen_and_translate()
    else:
        legacy_listen_and_translate() 