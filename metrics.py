"""
Advanced real-time metrics tracking for the speech-to-speech translation app.
Focuses on measuring speech-to-text, translation, and text-to-speech latencies.
"""
import time
import threading
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import statistics
import os
from tabulate import tabulate
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class LatencyMetrics:
    """Tracks detailed timing data for a single speech segment."""
    
    def __init__(self, segment_id: str):
        self.segment_id = segment_id
        self.speech_detected_time: Optional[float] = None
        self.stt_first_byte_time: Optional[float] = None
        self.stt_completed_time: Optional[float] = None
        self.translation_start_time: Optional[float] = None
        self.translation_first_token_time: Optional[float] = None
        self.translation_completed_time: Optional[float] = None
        self.tts_start_time: Optional[float] = None
        self.tts_first_audio_byte_time: Optional[float] = None
        self.tts_completed_time: Optional[float] = None
        
        # Additional streaming-specific metrics
        self.utterance_end_time: Optional[float] = None  # When the user stops speaking
        self.stt_first_interim_time: Optional[float] = None  # First interim result
        self.stt_first_final_time: Optional[float] = None  # First final result
        self.translation_first_meaningful_time: Optional[float] = None  # First meaningful translation chunk
        self.translation_token_count: int = 0  # Count of tokens generated
        self.translation_last_token_time: Optional[float] = None  # Time of last token for rate calculation
        self.translation_token_rate: Optional[float] = None  # Token generation rate (tokens/second)
        
        # Audio processing metrics
        self.bytes_processed: int = 0  # Total bytes of audio processed
        
        # Derived metrics (calculated on demand)
        self._stt_first_byte_latency: Optional[float] = None
        self._translation_first_token_latency: Optional[float] = None
        self._tts_first_byte_latency: Optional[float] = None
        self._end_to_end_latency: Optional[float] = None
        
    @property
    def stt_first_byte_latency(self) -> Optional[float]:
        """Time from speech detected to first byte from STT (ms)."""
        if self.speech_detected_time and self.stt_first_byte_time:
            return (self.stt_first_byte_time - self.speech_detected_time) * 1000
        return None
    
    @property
    def stt_first_byte_latency_per_byte(self) -> Optional[float]:
        """Time from speech detected to first byte from STT per byte processed (ms/byte)."""
        if self.speech_detected_time and self.stt_first_byte_time and self.bytes_processed > 0:
            total_latency = (self.stt_first_byte_time - self.speech_detected_time) * 1000
            return total_latency / self.bytes_processed
        return None
    
    @property
    def stt_first_interim_latency(self) -> Optional[float]:
        """Time from speech detected to first interim result (ms)."""
        if self.speech_detected_time and self.stt_first_interim_time:
            return (self.stt_first_interim_time - self.speech_detected_time) * 1000
        return None
    
    @property
    def stt_first_interim_latency_per_byte(self) -> Optional[float]:
        """Time from speech detected to first interim result per byte processed (ms/byte)."""
        if self.speech_detected_time and self.stt_first_interim_time and self.bytes_processed > 0:
            total_latency = (self.stt_first_interim_time - self.speech_detected_time) * 1000
            return total_latency / self.bytes_processed
        return None
    
    @property
    def stt_first_final_latency(self) -> Optional[float]:
        """Time from speech detected to first final result (ms)."""
        if self.speech_detected_time and self.stt_first_final_time:
            return (self.stt_first_final_time - self.speech_detected_time) * 1000
        return None
    
    @property
    def stt_first_final_latency_per_byte(self) -> Optional[float]:
        """Time from speech detected to first final result per byte processed (ms/byte)."""
        if self.speech_detected_time and self.stt_first_final_time and self.bytes_processed > 0:
            total_latency = (self.stt_first_final_time - self.speech_detected_time) * 1000
            return total_latency / self.bytes_processed
        return None
    
    @property
    def utterance_to_final_latency(self) -> Optional[float]:
        """Time from end of utterance to final transcription (ms)."""
        if self.utterance_end_time and self.stt_completed_time:
            return (self.stt_completed_time - self.utterance_end_time) * 1000
        return None
        
    @property
    def translation_first_token_latency(self) -> Optional[float]:
        """Time from STT completion to first token from translation (ms)."""
        if self.stt_completed_time and self.translation_first_token_time:
            return (self.translation_first_token_time - self.stt_completed_time) * 1000
        return None
    
    @property
    def translation_first_meaningful_latency(self) -> Optional[float]:
        """Time from STT completion to first meaningful translation chunk (ms)."""
        if self.stt_completed_time and self.translation_first_meaningful_time:
            return (self.translation_first_meaningful_time - self.stt_completed_time) * 1000
        return None
    
    @property
    def translation_token_rate(self) -> Optional[float]:
        """Token generation rate (tokens per second)."""
        if (self.translation_first_token_time and self.translation_last_token_time and 
            self.translation_token_count > 0 and 
            self.translation_last_token_time > self.translation_first_token_time):
            duration = self.translation_last_token_time - self.translation_first_token_time
            return self.translation_token_count / duration if duration > 0 else 0
        return None
    
    @translation_token_rate.setter
    def translation_token_rate(self, value: float) -> None:
        """Set the token generation rate directly."""
        self._translation_token_rate = value
        
    @translation_token_rate.getter
    def translation_token_rate(self) -> Optional[float]:
        """Get the token generation rate."""
        if hasattr(self, '_translation_token_rate') and self._translation_token_rate is not None:
            return self._translation_token_rate
            
        # Fall back to calculated value
        if (self.translation_first_token_time and self.translation_last_token_time and 
            self.translation_token_count > 0 and 
            self.translation_last_token_time > self.translation_first_token_time):
            duration = self.translation_last_token_time - self.translation_first_token_time
            return self.translation_token_count / duration if duration > 0 else 0
        return None
        
    @property
    def tts_first_byte_latency(self) -> Optional[float]:
        """Time from translation first token to first audio byte from TTS (ms)."""
        if self.translation_first_token_time and self.tts_first_audio_byte_time:
            return (self.tts_first_audio_byte_time - self.translation_first_token_time) * 1000
        return None
        
    @property
    def end_to_end_latency(self) -> Optional[float]:
        """Time from speech detected to first audio byte from TTS (ms)."""
        if self.speech_detected_time and self.tts_first_audio_byte_time:
            return (self.tts_first_audio_byte_time - self.speech_detected_time) * 1000
        return None
    
    @property
    def utterance_to_audio_latency(self) -> Optional[float]:
        """Time from end of utterance to first audio byte from TTS (ms)."""
        if self.utterance_end_time and self.tts_first_audio_byte_time:
            return (self.tts_first_audio_byte_time - self.utterance_end_time) * 1000
        return None
    
    def as_dict(self) -> Dict[str, Any]:
        """Return metrics as a dictionary for tabulation or storage."""
        return {
            "segment_id": self.segment_id,
            "speech_detected": self.speech_detected_time,
            "stt_first_byte": self.stt_first_byte_latency,
            "stt_first_byte_per_byte": self.stt_first_byte_latency_per_byte,
            "stt_first_interim": self.stt_first_interim_latency,
            "stt_first_interim_per_byte": self.stt_first_interim_latency_per_byte,
            "stt_first_final": self.stt_first_final_latency,
            "stt_first_final_per_byte": self.stt_first_final_latency_per_byte,
            "utterance_to_final": self.utterance_to_final_latency,
            "translation_first_token": self.translation_first_token_latency,
            "translation_first_meaningful": self.translation_first_meaningful_latency,
            "translation_token_rate": self.translation_token_rate,
            "tts_first_audio_byte": self.tts_first_byte_latency,
            "end_to_end": self.end_to_end_latency,
            "utterance_to_audio": self.utterance_to_audio_latency,
            "bytes_processed": self.bytes_processed
        }

class MetricsTracker:
    """
    Advanced metrics tracking for the speech-to-speech translation process.
    Focuses on measuring latency from speech detection to first responses.
    """
    def __init__(self):
        # Core metrics data storage
        self.latency_metrics: Dict[str, LatencyMetrics] = {}
        self.active_speech_segments: Dict[str, str] = {}  # Maps stream_id to segment_id
        
        # Latency metrics collections
        self.stt_first_byte_latencies: List[float] = []
        self.stt_first_byte_per_byte_latencies: List[float] = []  # New per-byte metric
        self.stt_first_interim_latencies: List[float] = []
        self.stt_first_interim_per_byte_latencies: List[float] = []  # New per-byte metric
        self.stt_first_final_latencies: List[float] = []
        self.stt_first_final_per_byte_latencies: List[float] = []  # New per-byte metric
        self.utterance_to_final_latencies: List[float] = []
        self.translation_first_token_latencies: List[float] = []
        self.translation_first_meaningful_latencies: List[float] = []
        self.translation_token_rates: List[float] = []
        self.tts_first_byte_latencies: List[float] = []
        self.end_to_end_latencies: List[float] = []
        self.utterance_to_audio_latencies: List[float] = []
        
        # Throughput metrics
        self.total_audio_processed: int = 0  # Total bytes of audio processed
        self.total_audio_segments: int = 0  # Total number of speech segments
        
        # Session info
        self.start_time = time.time()
        self.last_report_time = time.time()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Auto reporting
        self.auto_report_interval = 30  # seconds
        self._auto_report_thread = None
        self._running = False
        
        # Session ID
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory if it doesn't exist
        os.makedirs("metrics", exist_ok=True)
        
        # Backward compatibility - dictionary to store active timers
        self.timers = {}
        
        # Backward compatibility - old style metrics collections
        self.transcription_times = []
        self.translation_times = []
        self.tts_times = []
        self.total_translations = 0
    
    def start_auto_reporting(self):
        """Start automatic reporting at regular intervals."""
        self._running = True
        self._auto_report_thread = threading.Thread(target=self._auto_report_worker)
        self._auto_report_thread.daemon = True
        self._auto_report_thread.start()
        
    def stop_auto_reporting(self):
        """Stop automatic reporting."""
        self._running = False
        if self._auto_report_thread:
            self._auto_report_thread.join(timeout=1.0)
    
    def _auto_report_worker(self):
        """Worker thread for automatic reporting."""
        while self._running:
            time.sleep(self.auto_report_interval)
            self.report()
    
    def track_speech_detected(self, stream_id: str) -> str:
        """Track when speech is detected from the microphone."""
        with self._lock:
            # Generate a unique segment ID
            segment_id = f"segment_{time.time()}_{stream_id}"
            
            # Create new latency metrics object
            metrics = LatencyMetrics(segment_id)
            metrics.speech_detected_time = time.time()
            
            # Store metrics and map stream to segment
            self.latency_metrics[segment_id] = metrics
            self.active_speech_segments[stream_id] = segment_id
            
            self.total_audio_segments += 1
            
            # Start the STT timer when speech is detected
            self.start_timer("stt")
            
            return segment_id
    
    def track_mic_activity(self, stream_id: str) -> str:
        """
        Track when microphone activity is detected.
        This is an earlier indicator than speech detection and provides more accurate timing.
        """
        with self._lock:
            # Generate a unique segment ID
            segment_id = f"segment_{time.time()}_{stream_id}"
            
            # Create new latency metrics object
            metrics = LatencyMetrics(segment_id)
            metrics.speech_detected_time = time.time()  # Use the same field for compatibility
            
            # Store metrics and map stream to segment
            self.latency_metrics[segment_id] = metrics
            self.active_speech_segments[stream_id] = segment_id
            
            self.total_audio_segments += 1
            
            # Start the STT timer when mic activity is detected
            self.start_timer("stt")
            
            return segment_id
    
    def track_stt_first_byte(self, stream_id: str) -> None:
        """Track when the first transcription byte is received from STT."""
        with self._lock:
            segment_id = self.active_speech_segments.get(stream_id)
            if not segment_id:
                return
                
            metrics = self.latency_metrics.get(segment_id)
            if metrics and not metrics.stt_first_byte_time:
                metrics.stt_first_byte_time = time.time()
                
                # If we have both start and first byte times, add to latency list
                if metrics.speech_detected_time:
                    latency = metrics.stt_first_byte_latency
                    if latency is not None:
                        self.stt_first_byte_latencies.append(latency)
                    
                    # Add per-byte latency if we have bytes processed
                    per_byte_latency = metrics.stt_first_byte_latency_per_byte
                    if per_byte_latency is not None:
                        self.stt_first_byte_per_byte_latencies.append(per_byte_latency)
    
    def track_stt_completed(self, stream_id: str) -> Optional[str]:
        """Track when STT transcription is complete for a segment."""
        with self._lock:
            segment_id = self.active_speech_segments.get(stream_id)
            if not segment_id:
                return None
                
            metrics = self.latency_metrics.get(segment_id)
            if metrics:
                metrics.stt_completed_time = time.time()
                return segment_id
            return None
    
    def track_translation_start(self, segment_id: str) -> None:
        """Track when translation starts for a segment."""
        with self._lock:
            metrics = self.latency_metrics.get(segment_id)
            if metrics:
                metrics.translation_start_time = time.time()
    
    def track_translation_first_token(self, segment_id: str) -> None:
        """Track when the first token is received from translation."""
        with self._lock:
            metrics = self.latency_metrics.get(segment_id)
            if metrics and not metrics.translation_first_token_time:
                metrics.translation_first_token_time = time.time()
                
                # If we have both start and first token times, add to latency list
                if metrics.stt_completed_time:
                    latency = metrics.translation_first_token_latency
                    if latency is not None:
                        self.translation_first_token_latencies.append(latency)
    
    def track_translation_completed(self, segment_id: str) -> None:
        """Track when translation is complete for a segment."""
        with self._lock:
            metrics = self.latency_metrics.get(segment_id)
            if metrics:
                metrics.translation_completed_time = time.time()
    
    def track_tts_start(self, segment_id: str) -> None:
        """Track when TTS synthesis starts for a segment."""
        with self._lock:
            metrics = self.latency_metrics.get(segment_id)
            if metrics:
                metrics.tts_start_time = time.time()
    
    def track_tts_first_audio_byte(self, segment_id: str) -> None:
        """
        Track when the first audio byte is received from TTS.
        
        Args:
            segment_id: The segment ID to track
        """
        with self._lock:
            metrics = self.latency_metrics.get(segment_id)
            if metrics and not metrics.tts_first_audio_byte_time:
                # Note: The actual timestamp may be updated externally for more accuracy
                metrics.tts_first_audio_byte_time = time.time()
                
                # Calculate and store latencies
                if metrics.translation_first_token_time:
                    latency = metrics.tts_first_byte_latency
                    if latency is not None:
                        self.tts_first_byte_latencies.append(latency)
                        logger.debug(f"TTS first byte latency: {latency:.2f}ms for segment {segment_id}")
                
                # Calculate end-to-end latency
                if metrics.speech_detected_time:
                    latency = metrics.end_to_end_latency
                    if latency is not None:
                        self.end_to_end_latencies.append(latency)
                        logger.debug(f"End-to-end latency: {latency:.2f}ms for segment {segment_id}")
                        
                # Log all timing data for this segment for debugging
                # Create formatted strings for each value before using them in the f-string
                stt_first_byte_str = f"{metrics.stt_first_byte_time:.6f}" if metrics.stt_first_byte_time else "None"
                translation_first_token_str = f"{metrics.translation_first_token_time:.6f}" if metrics.translation_first_token_time else "None"
                tts_first_audio_byte_str = f"{metrics.tts_first_audio_byte_time:.6f}" if metrics.tts_first_audio_byte_time else "None"
                
                logger.debug(f"Segment {segment_id} timing data: speech_detected={metrics.speech_detected_time:.6f}, " +
                           f"stt_first_byte={stt_first_byte_str}, " +
                           f"translation_first_token={translation_first_token_str}, " +
                           f"tts_first_audio_byte={tts_first_audio_byte_str}")
                
                # Log the complete latency breakdown
                self._log_latency_breakdown(metrics)
    
    def _log_latency_breakdown(self, metrics: LatencyMetrics) -> None:
        """
        Log a detailed breakdown of latency components for a segment.
        
        Args:
            metrics: The LatencyMetrics object to analyze
        """
        if not metrics.speech_detected_time or not metrics.tts_first_audio_byte_time:
            return
            
        # Calculate each component's contribution to total latency
        total_latency_ms = metrics.end_to_end_latency
        if total_latency_ms is None:
            return
            
        components = []
        
        # STT component
        if metrics.stt_first_byte_time:
            stt_latency_ms = (metrics.stt_first_byte_time - metrics.speech_detected_time) * 1000
            stt_percentage = (stt_latency_ms / total_latency_ms) * 100
            components.append(f"STT: {stt_latency_ms:.2f}ms ({stt_percentage:.1f}%)")
        
        # Translation component
        if metrics.stt_completed_time and metrics.translation_first_token_time:
            trans_latency_ms = (metrics.translation_first_token_time - metrics.stt_completed_time) * 1000
            trans_percentage = (trans_latency_ms / total_latency_ms) * 100
            components.append(f"Translation: {trans_latency_ms:.2f}ms ({trans_percentage:.1f}%)")
        
        # TTS component
        if metrics.translation_first_token_time and metrics.tts_first_audio_byte_time:
            tts_latency_ms = (metrics.tts_first_audio_byte_time - metrics.translation_first_token_time) * 1000
            tts_percentage = (tts_latency_ms / total_latency_ms) * 100
            components.append(f"TTS: {tts_latency_ms:.2f}ms ({tts_percentage:.1f}%)")
        
        # Log the breakdown
        logger.info(f"Latency breakdown for segment {metrics.segment_id}: " +
                   f"Total: {total_latency_ms:.2f}ms | " + " | ".join(components))
                   
        # Log detailed timing data for debugging
        stt_first_byte_str = f"{metrics.stt_first_byte_time:.6f}" if metrics.stt_first_byte_time else "None"
        translation_first_token_str = f"{metrics.translation_first_token_time:.6f}" if metrics.translation_first_token_time else "None"
        tts_first_audio_byte_str = f"{metrics.tts_first_audio_byte_time:.6f}" if metrics.tts_first_audio_byte_time else "None"
        
        logger.debug(f"Segment {metrics.segment_id} timing data: speech_detected={metrics.speech_detected_time:.6f}, " +
                    f"stt_first_byte={stt_first_byte_str}, " +
                    f"translation_first_token={translation_first_token_str}, " +
                    f"tts_first_audio_byte={tts_first_audio_byte_str}")
    
    def track_tts_completed(self, segment_id: str) -> None:
        """Track when TTS synthesis is complete for a segment."""
        with self._lock:
            metrics = self.latency_metrics.get(segment_id)
            if metrics:
                metrics.tts_completed_time = time.time()
    
    def track_audio_processed(self, stream_id: str, bytes_count: int) -> None:
        """Track amount of audio processed in bytes."""
        with self._lock:
            self.total_audio_processed += bytes_count
            
            # Track bytes processed for the active segment
            segment_id = self.active_speech_segments.get(stream_id)
            if segment_id:
                metrics = self.latency_metrics.get(segment_id)
                if metrics:
                    metrics.bytes_processed += bytes_count
                    # Log every 10KB of data to avoid excessive logging
                    if metrics.bytes_processed % 10240 < bytes_count:
                        logger.debug(f"Audio processed for segment {segment_id}: {metrics.bytes_processed} bytes total")
    
    def complete_speech_segment(self, stream_id: str) -> None:
        """Mark a speech segment as complete and remove from active list."""
        with self._lock:
            segment_id = self.active_speech_segments.get(stream_id)
            if segment_id:
                # Log per-byte metrics before removing the segment
                metrics = self.latency_metrics.get(segment_id)
                if metrics and metrics.bytes_processed > 0:
                    self._log_per_byte_metrics(metrics)
                
                # Remove from active segments
                del self.active_speech_segments[stream_id]
    
    def _log_per_byte_metrics(self, metrics: LatencyMetrics) -> None:
        """
        Log detailed per-byte metrics for a segment.
        
        Args:
            metrics: The LatencyMetrics object to analyze
        """
        if metrics.bytes_processed == 0:
            return
            
        # Log per-byte metrics
        logger.info(f"Per-byte metrics for segment {metrics.segment_id} (total bytes: {metrics.bytes_processed}):")
        
        if metrics.stt_first_byte_latency_per_byte is not None:
            logger.info(f"  STT first byte: {metrics.stt_first_byte_latency_per_byte:.6f} ms/byte")
            
        if metrics.stt_first_interim_latency_per_byte is not None:
            logger.info(f"  STT first interim: {metrics.stt_first_interim_latency_per_byte:.6f} ms/byte")
            
        if metrics.stt_first_final_latency_per_byte is not None:
            logger.info(f"  STT first final: {metrics.stt_first_final_latency_per_byte:.6f} ms/byte")
    
    def _calculate_statistics(self, data: List[float]) -> Dict[str, float]:
        """Calculate descriptive statistics for a list of values."""
        if not data:
            return {
                "min": 0,
                "max": 0,
                "mean": 0,
                "median": 0,
                "p95": 0,
                "count": 0
            }
            
        return {
            "min": min(data),
            "max": max(data),
            "mean": statistics.mean(data),
            "median": statistics.median(data),
            "p95": statistics.quantiles(data, n=20)[-1] if len(data) >= 20 else max(data),
            "count": len(data)
        }
    
    def get_summary_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all latency metrics."""
        with self._lock:
            return {
                "stt_first_byte": self._calculate_statistics(self.stt_first_byte_latencies),
                "stt_first_byte_per_byte": self._calculate_statistics(self.stt_first_byte_per_byte_latencies),
                "stt_first_interim": self._calculate_statistics(self.stt_first_interim_latencies),
                "stt_first_interim_per_byte": self._calculate_statistics(self.stt_first_interim_per_byte_latencies),
                "stt_first_final": self._calculate_statistics(self.stt_first_final_latencies),
                "stt_first_final_per_byte": self._calculate_statistics(self.stt_first_final_per_byte_latencies),
                "utterance_to_final": self._calculate_statistics(self.utterance_to_final_latencies),
                "translation_first_token": self._calculate_statistics(self.translation_first_token_latencies),
                "translation_first_meaningful": self._calculate_statistics(self.translation_first_meaningful_latencies),
                "translation_token_rate": self._calculate_statistics(self.translation_token_rates),
                "tts_first_byte": self._calculate_statistics(self.tts_first_byte_latencies),
                "end_to_end": self._calculate_statistics(self.end_to_end_latencies),
                "utterance_to_audio": self._calculate_statistics(self.utterance_to_audio_latencies)
            }
    
    def get_recent_segments(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get the most recent speech segments with their metrics."""
        with self._lock:
            # Sort segments by speech detected time (newest first)
            segments = sorted(
                [metrics for metrics in self.latency_metrics.values() if metrics.speech_detected_time],
                key=lambda m: m.speech_detected_time or 0,
                reverse=True
            )
            
            # Take the requested number of segments
            return [segment.as_dict() for segment in segments[:count]]
    
    def export_to_csv(self, filename: Optional[str] = None) -> str:
        """Export all metrics to a CSV file."""
        if filename is None:
            filename = f"metrics/latency_metrics_{self.session_id}.csv"
            
        with self._lock:
            # Convert all metrics to a list of dictionaries
            data = [metrics.as_dict() for metrics in self.latency_metrics.values()]
            
            # Create a DataFrame and export to CSV
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            
            return filename
    
    def report(self) -> None:
        """Display a comprehensive report of collected metrics."""
        runtime = time.time() - self.start_time
        interval = time.time() - self.last_report_time
        self.last_report_time = time.time()
        
        stats = self.get_summary_statistics()
        recent_segments = self.get_recent_segments(5)
        
        # Print header
        print("\n" + "="*80)
        print(f"SPEECH TRANSLATION METRICS REPORT - Session {self.session_id}")
        print(f"Runtime: {runtime:.2f} seconds | Interval: {interval:.2f} seconds")
        print("="*80)
        
        # Print summary statistics - organized by component
        print("\nSPEECH-TO-TEXT LATENCY (milliseconds):")
        stt_table = []
        stt_metrics = {
            "stt_first_byte": "First Byte",
            "stt_first_interim": "First Interim Result",
            "stt_first_final": "First Final Result",
            "utterance_to_final": "Utterance End → Final"
        }
        
        for metric_key, metric_name in stt_metrics.items():
            metric_data = stats[metric_key]
            stt_table.append([
                metric_name,
                f"{metric_data['min']:.1f}",
                f"{metric_data['median']:.1f}",
                f"{metric_data['mean']:.1f}",
                f"{metric_data['p95']:.1f}",
                f"{metric_data['max']:.1f}",
                metric_data['count']
            ])
        
        print(tabulate(
            stt_table,
            headers=["Metric", "Min", "Median", "Mean", "P95", "Max", "Count"],
            tablefmt="grid"
        ))
        
        # Add per-byte metrics table
        print("\nSPEECH-TO-TEXT LATENCY PER BYTE (milliseconds/byte):")
        stt_byte_table = []
        stt_byte_metrics = {
            "stt_first_byte_per_byte": "First Byte per Byte",
            "stt_first_interim_per_byte": "First Interim Result per Byte",
            "stt_first_final_per_byte": "First Final Result per Byte"
        }
        
        for metric_key, metric_name in stt_byte_metrics.items():
            metric_data = stats[metric_key]
            stt_byte_table.append([
                metric_name,
                f"{metric_data['min']:.6f}",
                f"{metric_data['median']:.6f}",
                f"{metric_data['mean']:.6f}",
                f"{metric_data['p95']:.6f}",
                f"{metric_data['max']:.6f}",
                metric_data['count']
            ])
        
        print(tabulate(
            stt_byte_table,
            headers=["Metric", "Min", "Median", "Mean", "P95", "Max", "Count"],
            tablefmt="grid"
        ))
        
        print("\nTRANSLATION LATENCY (milliseconds):")
        translation_table = []
        translation_metrics = {
            "translation_first_token": "First Token",
            "translation_first_meaningful": "First Meaningful Chunk",
            "translation_token_rate": "Token Rate (tokens/sec)"
        }
        
        for metric_key, metric_name in translation_metrics.items():
            metric_data = stats[metric_key]
            translation_table.append([
                metric_name,
                f"{metric_data['min']:.1f}",
                f"{metric_data['median']:.1f}",
                f"{metric_data['mean']:.1f}",
                f"{metric_data['p95']:.1f}",
                f"{metric_data['max']:.1f}",
                metric_data['count']
            ])
        
        print(tabulate(
            translation_table,
            headers=["Metric", "Min", "Median", "Mean", "P95", "Max", "Count"],
            tablefmt="grid"
        ))
        
        print("\nTEXT-TO-SPEECH LATENCY (milliseconds):")
        tts_table = []
        tts_metrics = {
            "tts_first_byte": "First Audio Byte"
        }
        
        for metric_key, metric_name in tts_metrics.items():
            metric_data = stats[metric_key]
            tts_table.append([
                metric_name,
                f"{metric_data['min']:.1f}",
                f"{metric_data['median']:.1f}",
                f"{metric_data['mean']:.1f}",
                f"{metric_data['p95']:.1f}",
                f"{metric_data['max']:.1f}",
                metric_data['count']
            ])
        
        print(tabulate(
            tts_table,
            headers=["Metric", "Min", "Median", "Mean", "P95", "Max", "Count"],
            tablefmt="grid"
        ))
        
        print("\nEND-TO-END LATENCY (milliseconds):")
        e2e_table = []
        e2e_metrics = {
            "end_to_end": "Speech → Audio",
            "utterance_to_audio": "Utterance End → Audio"
        }
        
        for metric_key, metric_name in e2e_metrics.items():
            metric_data = stats[metric_key]
            e2e_table.append([
                metric_name,
                f"{metric_data['min']:.1f}",
                f"{metric_data['median']:.1f}",
                f"{metric_data['mean']:.1f}",
                f"{metric_data['p95']:.1f}",
                f"{metric_data['max']:.1f}",
                metric_data['count']
            ])
        
        print(tabulate(
            e2e_table,
            headers=["Metric", "Min", "Median", "Mean", "P95", "Max", "Count"],
            tablefmt="grid"
        ))
        
        # Print recent segments - simplified view with key metrics
        print("\nRECENT SPEECH SEGMENTS (milliseconds):")
        segments_table = []
        
        for segment in recent_segments:
            segments_table.append([
                segment["segment_id"].split("_")[1][:8],
                f"{segment['bytes_processed']}" if segment['bytes_processed'] else "0",
                f"{segment['stt_first_byte']:.1f}" if segment['stt_first_byte'] else "-",
                f"{segment['stt_first_byte_per_byte']:.6f}" if segment['stt_first_byte_per_byte'] else "-",
                f"{segment['translation_first_token']:.1f}" if segment['translation_first_token'] else "-",
                f"{segment['tts_first_audio_byte']:.1f}" if segment['tts_first_audio_byte'] else "-",
                f"{segment['end_to_end']:.1f}" if segment['end_to_end'] else "-"
            ])
        
        print(tabulate(
            segments_table,
            headers=["Time", "Bytes", "STT (ms)", "STT/byte", "Translation", "TTS", "End-to-End"],
            tablefmt="grid"
        ))
        
        # Print throughput statistics
        print("\nTHROUGHPUT STATISTICS:")
        mb_processed = self.total_audio_processed / 1024 / 1024
        print(f"Total audio processed: {mb_processed:.2f} MB")
        print(f"Total speech segments: {self.total_audio_segments}")
        if runtime > 0:
            print(f"Average audio processing rate: {mb_processed / (runtime / 60):.2f} MB/minute")
            print(f"Average segments per minute: {self.total_audio_segments / (runtime / 60):.2f}")
        
        # Include legacy metrics
        print("\nLEGACY METRICS:")
        print(f"Total translations: {self.total_translations}")
        
        if self.transcription_times:
            print(f"Avg transcription time: {self.avg(self.transcription_times):.2f} ms")
        
        if self.translation_times:
            print(f"Avg translation time: {self.avg(self.translation_times):.2f} ms")
        
        if self.tts_times:
            print(f"Avg TTS time: {self.avg(self.tts_times):.2f} ms")
        
        print("\nCSV export available at:", self.export_to_csv())
        print("="*80 + "\n")

    # ===== BACKWARD COMPATIBILITY METHODS =====
    
    def start_timer(self, name):
        """Backward compatibility: Start a timer with the given name."""
        self.timers[name] = time.time() * 1000  # Store in ms
    
    def stop_timer(self, name):
        """
        Backward compatibility: Stop a timer with the given name and return the elapsed time.
        Also track the time in the appropriate category.
        """
        if name not in self.timers:
            return 0
        
        elapsed = time.time() * 1000 - self.timers[name]
        
        # Store in the appropriate category
        if name == "stt":
            self.track_transcription(elapsed)
        elif name == "gemini":
            self.track_translation(elapsed)
        elif name == "tts":
            self.track_tts(elapsed)
        
        # Remove the timer
        del self.timers[name]
        
        return elapsed
    
    def track_transcription(self, duration_ms):
        """Backward compatibility: Track time for speech-to-text transcription."""
        self.transcription_times.append(duration_ms)
    
    def track_translation(self, duration_ms):
        """Backward compatibility: Track time for text translation."""
        self.translation_times.append(duration_ms)
        self.total_translations += 1
    
    def track_tts(self, duration_ms):
        """Backward compatibility: Track time for text-to-speech synthesis."""
        self.tts_times.append(duration_ms)
    
    def avg(self, times):
        """Backward compatibility: Calculate average time."""
        return sum(times) / len(times) if times else 0
    
    def display(self):
        """Backward compatibility: Display collected metrics."""
        # Just redirect to our new report method
        self.report()
    
    # ===== END BACKWARD COMPATIBILITY METHODS =====

    def track_utterance_end(self, stream_id: str) -> None:
        """Track when the user stops speaking (end of utterance)."""
        with self._lock:
            segment_id = self.active_speech_segments.get(stream_id)
            if not segment_id:
                return
                
            metrics = self.latency_metrics.get(segment_id)
            if metrics:
                metrics.utterance_end_time = time.time()
                logger.debug(f"Utterance end detected for segment {segment_id}")

    def track_stt_interim_result(self, stream_id: str) -> None:
        """Track when the first interim transcription result is received."""
        with self._lock:
            segment_id = self.active_speech_segments.get(stream_id)
            if not segment_id:
                return
                
            metrics = self.latency_metrics.get(segment_id)
            if metrics and not metrics.stt_first_interim_time:
                metrics.stt_first_interim_time = time.time()
                
                # If we have both start and first interim times, add to latency list
                if metrics.speech_detected_time:
                    latency = metrics.stt_first_interim_latency
                    if latency is not None:
                        self.stt_first_interim_latencies.append(latency)
                        logger.debug(f"First interim result latency: {latency:.2f}ms for segment {segment_id}")
                    
                    # Add per-byte latency if we have bytes processed
                    per_byte_latency = metrics.stt_first_interim_latency_per_byte
                    if per_byte_latency is not None:
                        self.stt_first_interim_per_byte_latencies.append(per_byte_latency)
                        logger.debug(f"First interim result latency per byte: {per_byte_latency:.6f}ms/byte for segment {segment_id}")

    def track_stt_final_result(self, stream_id: str) -> None:
        """Track when the first final transcription result is received."""
        with self._lock:
            segment_id = self.active_speech_segments.get(stream_id)
            if not segment_id:
                return
                
            metrics = self.latency_metrics.get(segment_id)
            if metrics and not metrics.stt_first_final_time:
                metrics.stt_first_final_time = time.time()
                
                # If we have both start and first final times, add to latency list
                if metrics.speech_detected_time:
                    latency = metrics.stt_first_final_latency
                    if latency is not None:
                        self.stt_first_final_latencies.append(latency)
                        logger.debug(f"First final result latency: {latency:.2f}ms for segment {segment_id}")
                    
                    # Add per-byte latency if we have bytes processed
                    per_byte_latency = metrics.stt_first_final_latency_per_byte
                    if per_byte_latency is not None:
                        self.stt_first_final_per_byte_latencies.append(per_byte_latency)
                        logger.debug(f"First final result latency per byte: {per_byte_latency:.6f}ms/byte for segment {segment_id}")
                
                # If we have utterance end time, calculate utterance-to-final latency
                if metrics.utterance_end_time:
                    latency = metrics.utterance_to_final_latency
                    if latency is not None:
                        self.utterance_to_final_latencies.append(latency)
                        logger.debug(f"Utterance to final result latency: {latency:.2f}ms for segment {segment_id}")

    def track_translation_meaningful_chunk(self, segment_id: str) -> None:
        """Track when the first meaningful translation chunk is received."""
        with self._lock:
            metrics = self.latency_metrics.get(segment_id)
            if metrics and not metrics.translation_first_meaningful_time:
                metrics.translation_first_meaningful_time = time.time()
                
                # If we have both start and first meaningful chunk times, add to latency list
                if metrics.stt_completed_time:
                    latency = metrics.translation_first_meaningful_latency
                    if latency is not None:
                        self.translation_first_meaningful_latencies.append(latency)
                        logger.debug(f"First meaningful translation chunk latency: {latency:.2f}ms for segment {segment_id}")

    def track_translation_token(self, segment_id: str) -> None:
        """Track when a translation token is received and update token count."""
        with self._lock:
            metrics = self.latency_metrics.get(segment_id)
            if metrics:
                metrics.translation_token_count += 1
                metrics.translation_last_token_time = time.time()
                
                # If we have enough data to calculate token rate, do it
                if (metrics.translation_first_token_time and 
                    metrics.translation_token_count > 5 and  # Need enough tokens for meaningful rate
                    metrics.translation_last_token_time > metrics.translation_first_token_time):
                    
                    # Calculate token rate (tokens per second)
                    time_diff = metrics.translation_last_token_time - metrics.translation_first_token_time
                    if time_diff > 0:
                        token_rate = (metrics.translation_token_count - 1) / time_diff
                        # Store the token rate in the metrics
                        metrics.translation_token_rate = token_rate
                        
                        # Only track if we haven't already or if it's significantly different
                        if (not self.translation_token_rates or 
                            len(self.translation_token_rates) == 0 or
                            abs(token_rate - self.translation_token_rates[-1]) / max(0.1, self.translation_token_rates[-1]) > 0.1):
                            self.translation_token_rates.append(token_rate)
                            logger.debug(f"Translation token rate: {token_rate:.2f} tokens/sec for segment {segment_id}")

# Create a singleton instance
tracker = MetricsTracker() 