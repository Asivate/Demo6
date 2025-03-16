"""
Helper functions for SoundWatch server.
"""

import easing
import sys
import time
import numpy as np
import wget
from math import log
import math

class Interpolator():
    def __init__(self, fps=60.0):
        self.start = 0.0
        self.end = 0.0
        self.values = []
        self.duration = 1.0
        self.easing = None
        self.start = 0
        self.x = []
        self.fps = 1.0/fps
        self.start_time = None
        
    def animate(self, f, t, d):
        self.start = f
        self.end = t
        self.duration = d
        self.start_time = time.time()
        self.easing = easing.QuadEaseInOut(start=self.start, end=self.end, duration=self.duration)
        self.x = np.arange(0, 1, self.fps)
        self.values = list(map(self.easing.ease, self.x))
        
    def update(self):
        if (self.start_time is not None):
            # find index based on current time diff between start-time + duration
            diff = time.time() - self.start_time
            index = min(len(self.values)-1,int(diff / self.fps))
            return(self.values[index])
        return 0.0
        
def ratio_to_db(ratio, val2=None, using_amplitude=True):
    ratio = float(ratio)
    # accept 2 values and use the ratio of val1 to val2
    if val2 is not None:
        ratio = ratio / val2

    # special case for multiply-by-zero (convert to silence)
    if ratio == 0:
        return -float('inf')

    if using_amplitude:
        return 20 * log(ratio, 10)
    else:  # using power
        return 10 * log(ratio, 10)
        
def dbFS(audio_data):
    """
    Calculate the dB level of an audio signal.
    
    Args:
        audio_data: Numpy array of audio samples
        
    Returns:
        float: dB level of the audio signal
    """
    if len(audio_data) == 0:
        return -100.0  # Return very low dB for empty audio
        
    # Calculate RMS value
    rms = np.sqrt(np.mean(np.square(audio_data)))
    
    # Avoid log of zero
    if rms < 1e-10:
        return -100.0
        
    # Calculate dB
    db = 20 * math.log10(rms)
    return db

def rangemap(value, istart, istop, ostart, ostop):
    return ostart + (ostop - ostart) * ((value - istart) / (istop - istart))

def normalize_audio(audio_data):
    """
    Normalize audio data to range [-1, 1].
    
    Args:
        audio_data: Numpy array of audio samples
        
    Returns:
        Numpy array: Normalized audio data
    """
    if len(audio_data) == 0:
        return audio_data
        
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        return audio_data / max_val
    return audio_data

def get_audio_duration(sample_rate, num_samples):
    """
    Calculate the duration of an audio clip in seconds.
    
    Args:
        sample_rate: Sample rate in Hz
        num_samples: Number of audio samples
        
    Returns:
        float: Duration in seconds
    """
    return num_samples / sample_rate

def format_time(seconds):
    """
    Format time in seconds to a human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        str: Formatted time string (MM:SS)
    """
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"
