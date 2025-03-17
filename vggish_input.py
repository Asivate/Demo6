# MFCC Spectrogram conversion code from VGGish, Google Inc.
# https://github.com/tensorflow/models/tree/master/research/audioset

import numpy as np
from scipy.io import wavfile
import mel_features
import vggish_params

def waveform_to_examples(data, sample_rate):
  # Convert to mono.
  if len(data.shape) > 1:
    data = np.mean(data, axis=1)
    
  # Check if data is too short for feature extraction
  min_samples = int(vggish_params.EXAMPLE_WINDOW_SECONDS * vggish_params.SAMPLE_RATE)
  if len(data) < min_samples:
    print(f"Warning: Audio data is too short ({len(data)} samples), minimum required is {min_samples} samples.")
    # Return empty array if too short (caller will handle this case)
    return np.array([])

  # Compute log mel spectrogram features.
  try:
    log_mel = mel_features.log_mel_spectrogram(
        data,
        audio_sample_rate=vggish_params.SAMPLE_RATE,
        log_offset=vggish_params.LOG_OFFSET,
        window_length_secs=vggish_params.STFT_WINDOW_LENGTH_SECONDS,
        hop_length_secs=vggish_params.STFT_HOP_LENGTH_SECONDS,
        num_mel_bins=vggish_params.NUM_MEL_BINS,
        lower_edge_hertz=vggish_params.MEL_MIN_HZ,
        upper_edge_hertz=vggish_params.MEL_MAX_HZ)

    # Frame features into examples.
    features_sample_rate = 1.0 / vggish_params.STFT_HOP_LENGTH_SECONDS
    example_window_length = int(round(
        vggish_params.EXAMPLE_WINDOW_SECONDS * features_sample_rate))
    example_hop_length = int(round(
        vggish_params.EXAMPLE_HOP_SECONDS * features_sample_rate))
    log_mel_examples = mel_features.frame(
        log_mel,
        window_length=example_window_length,
        hop_length=example_hop_length)
    
    if log_mel_examples.shape[0] == 0:
      print("No examples were generated from the audio.")
    else:
      print(f"Successfully generated {log_mel_examples.shape[0]} examples from the audio.")
    
    return log_mel_examples
    
  except Exception as e:
    print(f"Error extracting audio features: {str(e)}")
    return np.array([])


def wavfile_to_examples(wav_file):
  sr, wav_data = wavfile.read(wav_file)
  assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
  samples = wav_data / 32768.0  # Convert to [-1.0, +1.0]
  return waveform_to_examples(samples, sr)
