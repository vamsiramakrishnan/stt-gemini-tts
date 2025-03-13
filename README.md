# Speech-to-Speech Translation System

A real-time speech translation system that uses Google Cloud services to convert speech in one language to speech in another language. The system captures audio input, transcribes it using Speech-to-Text, translates the text using Gemini AI, and synthesizes speech in the target language using Text-to-Speech.

## Architecture

This application follows a modular, event-driven architecture designed for real-time speech processing with minimal latency. The system is built around the following key components:

### Core Components

1. **Event Bus** (`modules/event_bus.py`)
   - Central communication hub that enables loose coupling between components
   - Supports both synchronous and asynchronous event handling
   - Allows components to publish and subscribe to events without direct dependencies

2. **Pipeline Service** (`modules/pipeline.py`)
   - Orchestrates the entire speech processing workflow
   - Manages the lifecycle of speech processing pipelines
   - Handles error recovery and performance optimization
   - Implements adaptive parameters based on system performance

3. **Speech-to-Text Service** (`modules/speech_to_text.py`)
   - Captures audio from microphone input
   - Performs voice activity detection
   - Streams audio to Google Cloud Speech-to-Text API
   - Processes transcription results

4. **Translation Service** (`modules/text_translation.py`)
   - Translates text between languages using Google's Gemini AI
   - Implements caching for improved performance
   - Handles context-aware translation

5. **Text-to-Speech Service** (`modules/text_to_speech.py`)
   - Converts translated text to speech using Google Cloud Text-to-Speech API
   - Manages voice selection based on target language
   - Streams audio output for real-time playback

6. **Metrics System** (`metrics.py`, `metrics_integration.py`)
   - Tracks performance metrics like latency, word count, and error rates
   - Generates reports for system performance analysis
   - Supports adaptive optimization based on performance data

### Data Flow

1. Audio is captured from the microphone in chunks
2. Speech-to-Text service processes the audio and detects speech segments
3. Transcribed text is published to the event bus
4. Translation service receives the transcription and translates it
5. Translated text is published to the event bus
6. Text-to-Speech service synthesizes speech from the translated text
7. Audio is played back to the user

## Configuration

### Environment Variables

The application uses the following environment variables:

```
GOOGLE_CLOUD_PROJECT=vital-octagon-19612
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_CLOUD_STT_LOCATION=global
```

### Configuration Settings

Key configuration settings are defined in `config.py`:

- **Audio Settings**:
  - Format: 16-bit PCM
  - Channels: 1 (mono)
  - Sample Rate: 16000 Hz (input), 24000 Hz (output)
  - Chunk Size: 1600 samples (100ms)
  - Streaming Limit: 60 seconds

- **Default Languages**:
  - Source Language: en-US (English)
  - Target Language: hi-IN (Hindi)

- **Google Cloud Settings**:
  - Project ID: vital-octagon-19612
  - Location: us-central1 (for Gemini AI)
  - STT Location: global (for Speech-to-Text)

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- Google Cloud account with the following APIs enabled:
  - Speech-to-Text API
  - Vertex AI (for Gemini)
  - Text-to-Speech API
- Google Cloud credentials configured

### Dependencies

The application requires the following main dependencies:

```
# Core dependencies
google-cloud-speech
google-cloud-texttospeech
google-generativeai
pyaudio

# Metrics dependencies
pandas>=1.3.0
tabulate>=0.8.9
numpy>=1.20.0
matplotlib>=3.4.0
statistics>=1.0.3.5
```

### Installation

1. Clone the repository
2. Set up a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   pip install -r requirements_metrics.txt
   ```
4. Set up Google Cloud credentials:
   ```
   export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/credentials.json"
   ```
5. Configure environment variables in `.env` file

## Usage

Run the application with:

```
python main.py
```

### Command-line Options

- `--source-lang`: Source language code (default: en-US)
- `--target-lang`: Target language code (default: hi-IN)
- `--metrics`: Enable detailed metrics collection
- `--debug`: Enable debug logging

## Performance Optimization

The system includes several performance optimizations:

1. **Adaptive Parameters**: The pipeline adjusts parameters based on system performance
2. **Caching**: Frequently used translations and TTS outputs are cached
3. **Asynchronous Processing**: Non-blocking event handling for improved responsiveness
4. **Voice Activity Detection**: Reduces unnecessary processing during silence
5. **Health Monitoring**: Automatic recovery from component failures

## Metrics and Monitoring

The metrics system tracks:

- End-to-end latency
- Component-specific latency (STT, translation, TTS)
- Word count and processing rate
- Error rates and types
- System resource utilization

Metrics are stored in the `metrics/` directory and can be analyzed using the included reporting tools.

## Troubleshooting

Common issues:

1. **Authentication Errors**: Ensure GOOGLE_APPLICATION_CREDENTIALS is set correctly
2. **Audio Device Issues**: Check microphone permissions and PyAudio configuration
3. **API Quota Limits**: Monitor Google Cloud usage and adjust quotas if needed
4. **High Latency**: Check network connection and consider adjusting chunk size

## License

[Specify your license here]

## Contributors

[List contributors here] 