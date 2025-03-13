#!/usr/bin/env python3
"""
Configuration module for the speech translation system.
"""

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

@dataclass
class SpeechToTextConfig:
    """Configuration for speech-to-text services."""
    project_id: str = ""
    model: str = "latest_short"
    language_code: str = "en-US"
    auto_punctuation: bool = True
    enable_voice_activity_events: bool = True
    vad_sensitivity: float = 0.5  # Between 0.0 and 1.0
    max_alternatives: int = 1
    profanity_filter: bool = False

@dataclass
class TextToSpeechConfig:
    """Configuration for text-to-speech services."""
    voice_map: Dict[str, str] = field(default_factory=lambda: {
        "es-US": "es-US-Chirp3-HD-D",
        "fr-FR": "fr-FR-Chirp3-HD-D", 
        "de-DE": "de-DE-Chirp-HD-D",
        "it-IT": "it-IT-Chirp-HD-D",
        "ja-JP": "ja-JP-Chirp3-HD-Kore",
        "hi-IN": "hi-IN-Chirp3-HD-Aoede"
    })
    default_voice: str = "es-US-Chirp3-HD-Kore"
    speaking_rate: float = 1.0
    pitch: float = 0.0
    volume_gain_db: float = 0.0
    enable_streaming: bool = True
    chunk_size: int = 1024

@dataclass
class TranslationConfig:
    """Configuration for translation services."""
    model: str = "gemini-2.0-flash-001"
    temperature: float = 0.2
    max_output_tokens: int = 512
    top_p: float = 0.8
    top_k: int = 40

@dataclass
class AppConfig:
    """Main application configuration."""
    stt_config: SpeechToTextConfig = field(default_factory=SpeechToTextConfig)
    tts_config: TextToSpeechConfig = field(default_factory=TextToSpeechConfig)
    translation_config: TranslationConfig = field(default_factory=TranslationConfig)
    default_source_lang: str = "en-US"
    default_target_lang: str = "es-US"
    audio_sample_rate: int = 16000
    enable_metrics: bool = True
    log_level: str = "INFO"
    max_translation_threads: int = 2
    echo_cancellation_enabled: bool = True
    echo_cancellation_delay_ms: int = 500

class ConfigManager:
    """
    Manages application configuration, allowing for loading from files
    and updating settings at runtime.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Optional path to a JSON configuration file
        """
        self.config = AppConfig()
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)
        
    def load_from_file(self, config_path: str) -> None:
        """
        Load configuration from a JSON file.
        
        Args:
            config_path: Path to the JSON configuration file
        """
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                
            # Update STT config
            if 'stt_config' in config_data:
                for key, value in config_data['stt_config'].items():
                    if hasattr(self.config.stt_config, key):
                        setattr(self.config.stt_config, key, value)
            
            # Update TTS config
            if 'tts_config' in config_data:
                for key, value in config_data['tts_config'].items():
                    if hasattr(self.config.tts_config, key):
                        setattr(self.config.tts_config, key, value)
            
            # Update Translation config
            if 'translation_config' in config_data:
                for key, value in config_data['translation_config'].items():
                    if hasattr(self.config.translation_config, key):
                        setattr(self.config.translation_config, key, value)
            
            # Update main app config
            for key, value in config_data.items():
                if key not in ('stt_config', 'tts_config', 'translation_config') and hasattr(self.config, key):
                    setattr(self.config, key, value)
                    
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
    
    def save_to_file(self, config_path: str) -> None:
        """
        Save the current configuration to a JSON file.
        
        Args:
            config_path: Path to save the configuration
        """
        try:
            # Convert config to dict for JSON serialization
            config_dict = {
                'stt_config': {k: v for k, v in vars(self.config.stt_config).items()},
                'tts_config': {k: v for k, v in vars(self.config.tts_config).items()},
                'translation_config': {k: v for k, v in vars(self.config.translation_config).items()},
            }
            
            # Add main config items
            for k, v in vars(self.config).items():
                if k not in ('stt_config', 'tts_config', 'translation_config'):
                    config_dict[k] = v
            
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
                
            logger.info(f"Saved configuration to {config_path}")
        except Exception as e:
            logger.error(f"Error saving configuration to {config_path}: {e}")
    
    def update_config(self, config_path: Optional[str] = None, **kwargs) -> None:
        """
        Update configuration with new values.
        
        Args:
            config_path: Optional path to save updated config
            **kwargs: Configuration values to update
        """
        for key, value in kwargs.items():
            if '.' in key:
                # Handle nested configuration (e.g., 'stt_config.language_code')
                section, attribute = key.split('.', 1)
                if hasattr(self.config, section) and hasattr(getattr(self.config, section), attribute):
                    setattr(getattr(self.config, section), attribute, value)
            elif hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Save to file if path provided
        if config_path:
            self.save_to_file(config_path)

# Create global instance
config_manager = ConfigManager() 