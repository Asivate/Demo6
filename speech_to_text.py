"""
Speech-to-Text Module for SoundWatch

This module provides speech recognition functionality to convert audio to text.
It uses a lightweight model for real-time transcription.
"""
import numpy as np
import torch
from transformers import pipeline
from threading import Lock
import scipy.signal as sps
import re
import traceback
import os
from scipy.fft import rfft
from scipy import stats

# Define filter_hallucinations function at the module level so tests can access it
def filter_hallucinations(text):
    """
    Filter out common hallucination patterns from transcribed text.
    
    Args:
        text: The transcribed text to filter
        
    Returns:
        Filtered text with hallucination patterns removed
    """
    if not text:
        return text
        
    # Check for repetitive patterns (a common hallucination issue)
    words = text.split()
    if len(words) >= 6:
        # Check for exact repetition of 3+ word phrases
        for i in range(len(words) - 5):
            phrase1 = " ".join(words[i:i+3])
            for j in range(i+3, len(words) - 2):
                phrase2 = " ".join(words[j:j+3])
                if phrase1.lower() == phrase2.lower():
                    # Found repetition, keep only the first occurrence
                    return " ".join(words[:j])
    
    # Remove common hallucination phrases
    common_hallucinations = [
        "thank you for watching",
        "thanks for watching", 
        "please subscribe",
        "like and subscribe",
        "toad",
        "toe",
        "toe the"
    ]
    
    for phrase in common_hallucinations:
        if phrase in text.lower():
            # Remove the hallucination phrase
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            text = pattern.sub("", text)
    
    # Clean up any resulting double spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class SpeechToText:
    _instance = None
    _lock = Lock()
    _is_initialized = False
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SpeechToText, cls).__new__(cls)
            return cls._instance
    
    def __init__(self):
        if not self._is_initialized:
            self._initialize()
    
    def _initialize(self):
        """
        Initialize the speech recognition model.
        """
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Initializing speech recognition model...")
            
            # First try with the more modern parameter style
            try:
                self.asr_pipeline = pipeline(
                    "automatic-speech-recognition",
                    model="openai/whisper-medium",
                    device=device
                )
                print(f"Speech recognition model loaded successfully on {device}")
            except Exception as e:
                # Fall back to older style initialization if needed
                print(f"Warning: Modern initialization failed, trying legacy mode: {e}")
                from transformers import WhisperProcessor, WhisperForConditionalGeneration
                
                processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
                model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium").to(device)
                
                # Create a custom pipeline function that mimics the pipeline API
                def custom_asr_pipeline(inputs, max_new_tokens=None):
                    if isinstance(inputs, dict):
                        audio = inputs["raw"]
                        sample_rate = inputs["sampling_rate"]
                    else:
                        audio = inputs
                        sample_rate = 16000
                    
                    input_features = processor(audio, sampling_rate=sample_rate, return_tensors="pt").input_features.to(device)
                    
                    # Generate tokens using the model
                    with torch.no_grad():
                        generated_ids = model.generate(
                            input_features, 
                            max_new_tokens=max_new_tokens if max_new_tokens else 128
                        )
                    
                    # Decode the generated tokens to text
                    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    return {"text": transcription}
                
                self.asr_pipeline = custom_asr_pipeline
                print(f"Speech recognition model loaded successfully in legacy mode on {device}")
                
            # Mark as initialized
            SpeechToText._is_initialized = True
            
        except Exception as e:
            error_msg = f"Error initializing speech recognition model: {str(e)}"
            print(error_msg)
            traceback.print_exc()
    
    def preprocess_audio(self, audio_data, sample_rate=16000):
        """
        Preprocess audio data to improve speech recognition quality.
        
        Args:
            audio_data (numpy.ndarray): The audio data as a numpy array
            sample_rate (int): The sample rate of the audio (default: 16000)
            
        Returns:
            numpy.ndarray: The preprocessed audio data
        """
        if len(audio_data) == 0:
            return audio_data
            
        # Step 1: Apply pre-emphasis filter to emphasize higher frequencies
        # This helps with clarity of consonants
        preemphasis = 0.98
        emphasized_audio = np.append(audio_data[0], audio_data[1:] - preemphasis * audio_data[:-1])
        
        # Step 2: Apply bandpass filter to focus on speech frequencies (250-3400 Hz)
        # This is the standard telephone speech range where most vocal content exists
        nyquist = sample_rate / 2.0
        low_cutoff = 250 / nyquist
        high_cutoff = 3400 / nyquist
        b, a = sps.butter(4, [low_cutoff, high_cutoff], btype='band')
        filtered_audio = sps.filtfilt(b, a, emphasized_audio)
        
        # Step 3: Implement noise reduction using spectral subtraction
        # This estimates noise from low-energy frames and subtracts it
        frame_length = int(sample_rate * 0.025)  # 25ms frames
        hop_length = int(sample_rate * 0.010)    # 10ms hop
        
        # Estimate noise profile from the lowest 10% energy frames
        denoised_audio = filtered_audio
        if len(filtered_audio) > frame_length:
            frames = []
            for i in range(0, len(filtered_audio) - frame_length, hop_length):
                frames.append(filtered_audio[i:i+frame_length])
            
            if frames:
                frames = np.array(frames)
                frame_energies = np.sum(frames**2, axis=1)
                low_energy_threshold = np.percentile(frame_energies, 10)
                noise_frames = frames[frame_energies <= low_energy_threshold]
                
                if len(noise_frames) > 0:
                    noise_profile = np.mean(np.abs(rfft(noise_frames)), axis=0)
                    
                    # Process each frame
                    processed_frames = []
                    for i in range(0, len(filtered_audio) - frame_length, hop_length):
                        frame = filtered_audio[i:i+frame_length]
                        frame_spectrum = rfft(frame)
                        frame_magnitude = np.abs(frame_spectrum)
                        frame_phase = np.angle(frame_spectrum)
                        
                        # Spectral subtraction with flooring
                        reduction_factor = 2.0  # Adjust based on noise level
                        frame_magnitude = np.maximum(frame_magnitude - reduction_factor * noise_profile, 
                                                   0.01 * frame_magnitude)
                        
                        # Reconstruct frame and add to processed frames
                        processed_frame = np.real(np.fft.irfft(frame_magnitude * np.exp(1j * frame_phase)))
                        processed_frames.append(processed_frame)
                    
                    # Overlap-add to reconstruct the signal
                    denoised_audio = np.zeros(len(filtered_audio))
                    for i, frame in enumerate(processed_frames):
                        start = i * hop_length
                        end = start + frame_length
                        if end <= len(denoised_audio):
                            denoised_audio[start:end] += frame * np.hanning(frame_length)
        
        # Step 4: Apply dynamic range compression to make speech more consistent
        # This brings up quiet parts and lowers loud parts
        def compress_dynamic_range(audio, threshold=0.3, ratio=2.0):
            # Simple compressor
            indices_above = np.abs(audio) > threshold
            indices_below = ~indices_above
            
            audio_compressed = audio.copy()
            # Compress only the samples above threshold
            audio_compressed[indices_above] = np.sign(audio[indices_above]) * (
                threshold + (np.abs(audio[indices_above]) - threshold) / ratio
            )
            
            # Normalize to original RMS level
            original_rms = np.sqrt(np.mean(audio**2))
            compressed_rms = np.sqrt(np.mean(audio_compressed**2))
            if compressed_rms > 0:
                gain = original_rms / compressed_rms
                audio_compressed *= gain
            
            return audio_compressed
            
        compressed_audio = compress_dynamic_range(denoised_audio)
        
        # Step 5: Final clipping to ensure values are within [-1, 1]
        final_audio = np.clip(compressed_audio, -1.0, 1.0)
        
        return final_audio

    def transcribe(self, audio_data, sample_rate=16000):
        """
        Transcribe audio data to text.
        
        Args:
            audio_data (numpy.ndarray): The audio data as a numpy array
            sample_rate (int): The sample rate of the audio (default: 16000)
            
        Returns:
            str: The transcribed text
        """
        if audio_data is None or len(audio_data) == 0:
            return ""
        
        try:
            # Step 1: Voice Activity Detection to ensure there's actual speech
            # Calculate energy features
            frame_length = int(0.025 * sample_rate)  # 25ms frame
            hop_length = int(0.010 * sample_rate)    # 10ms hop
            
            # Calculate frame energies
            frames = []
            for i in range(0, len(audio_data) - frame_length, hop_length):
                frames.append(audio_data[i:i+frame_length])
                
            if not frames:
                return ""
                
            frames = np.array(frames)
            frame_energies = np.sum(frames**2, axis=1)
            
            # Calculate energy statistics
            energy_mean = np.mean(frame_energies)
            energy_std = np.std(frame_energies)
            energy_threshold = energy_mean + 0.5 * energy_std
            
            # Count frames above threshold
            speech_frames = np.sum(frame_energies > energy_threshold)
            speech_ratio = speech_frames / len(frame_energies) if len(frame_energies) > 0 else 0
            
            # If less than 15% of frames contain potential speech, return empty
            if speech_ratio < 0.05:
                print(f"Insufficient speech detected: {speech_ratio:.2f} ratio of frames above threshold")
                return ""
            
            # Step 2: Check if audio is of sufficient quality for transcription
            rms = np.sqrt(np.mean(audio_data**2))
            if rms < 0.005:  # Very low audio level
                print("Audio level too low for reliable transcription")
                return ""
                
            # Step 3: Preprocess audio to improve speech recognition
            processed_audio = self.preprocess_audio(audio_data, sample_rate)
            
            # Step 4: Ensure minimum duration for meaningful transcription
            # Whisper works better with at least 1-2 seconds of audio
            min_samples = sample_rate * 2  # 2 seconds minimum
            if len(processed_audio) < min_samples:
                # Pad only if we detected speech but it's too short
                if speech_ratio >= 0.15:
                    # Use better padding method for short audio
                    pad_size = min_samples - len(processed_audio)
                    # Mirror padding is better for speech than zero or edge padding
                    processed_audio = np.pad(processed_audio, (0, pad_size), mode='reflect')
                    print(f"Extended audio from {len(audio_data)} to {len(processed_audio)} samples using mirror padding")
                else:
                    return ""
            
            print("Transcribing with enhanced audio processing...")
            
            # Step 5: Detect repetition patterns in the audio that often cause hallucinations
            def detect_repetitions(audio, sample_rate, threshold=0.8):
                # Check for silence or near-silence at the end (common cause of hallucinations)
                window_size = int(0.5 * sample_rate)  # 500ms window
                if len(audio) > window_size:
                    end_energy = np.mean(audio[-window_size:]**2)
                    overall_energy = np.mean(audio**2)
                    if end_energy < 0.1 * overall_energy:
                        # Trim trailing silence
                        for i in range(len(audio)-1, window_size, -1):
                            if np.mean(audio[i-window_size:i]**2) > 0.2 * overall_energy:
                                return audio[:i]
                return audio
                
            processed_audio = detect_repetitions(processed_audio, sample_rate)
            
            # Step 6: Transcribe with improved parameters for better quality
            with self._lock:
                # Optimal parameters to reduce hallucinations - use only parameters known to be supported
                result = self.asr_pipeline(
                    {"raw": processed_audio, "sampling_rate": sample_rate},
                    max_new_tokens=256            # Increased for longer utterances
                )
                
                if result and "text" in result:
                    transcription = result["text"].strip()
                    
                    # Step 7: Analyze and filter the transcription
                    # Check for common hallucination patterns and filter them out
                    transcription = filter_hallucinations(transcription)
                    
                    # Clean up the transcription
                    # Remove filler words and normalize whitespace
                    filler_words = ["um", "uh", "er", "ah", "like", "you know"]
                    for word in filler_words:
                        transcription = re.sub(r'\b' + word + r'\b', '', transcription, flags=re.IGNORECASE)
                    
                    # Check for common single-word false positives
                    common_false_positives = ["the", "a", "that", "and", "you", "i", "but", "to", "it", "we", "are"]
                    if transcription.lower() in common_false_positives:
                        print(f"Ignoring common false positive transcription: '{transcription}'")
                        return ""
                    
                    # Normalize whitespace
                    transcription = ' '.join(transcription.split())
                    
                    # Check if there are actual meaningful words
                    words = [w for w in transcription.split() if w.lower() not in common_false_positives]
                    if len(words) < 2:
                        print(f"Transcription has too few meaningful words: '{transcription}'")
                        return ""
                    
                    print(f"Transcribed text: {transcription}")
                    return transcription
                else:
                    print("No transcription found in result")
                    return ""
        except Exception as e:
            print(f"Error transcribing audio: {str(e)}")
            traceback.print_exc()  # Print full traceback for debugging
            return ""

# Initialize the speech-to-text processor (singleton)
def transcribe_audio(audio_data, sample_rate=16000):
    """Convenience function to transcribe audio without manually creating a SpeechToText instance"""
    processor = SpeechToText()
    return processor.transcribe(audio_data, sample_rate) 