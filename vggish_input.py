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

  # Print audio information for debugging
  print(f"Audio data: shape={data.shape}, min={data.min():.4f}, max={data.max():.4f}, rms={np.sqrt(np.mean(data**2)):.4f}")
  
  # Ensure we have enough samples (at least 1 second at 16kHz)
  min_samples = vggish_params.SAMPLE_RATE  # 16000 samples
  if len(data) < min_samples:
    print(f"Warning: Audio data length {len(data)} is less than 1 second ({min_samples} samples)")
    # Pad with zeros to reach 1 second if needed
    padding = np.zeros(min_samples - len(data))
    data = np.concatenate((data, padding))
    print(f"Padded audio to {len(data)} samples")

  # Compute log mel spectrogram features.
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
      
  # Check if we got any frames
  if log_mel_examples.shape[0] == 0:
    print("Warning: No frames were generated. Creating dummy frame.")
    # Create a dummy frame with the right dimensions
    log_mel_examples = np.zeros((1, vggish_params.NUM_FRAMES, vggish_params.NUM_BANDS))
  
  print(f"Generated features: shape={log_mel_examples.shape}")
  return log_mel_examples


def wavfile_to_examples(wav_file):
  sr, wav_data = wavfile.read(wav_file)
  assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
  samples = wav_data / 32768.0  # Convert to [-1.0, +1.0]
  return waveform_to_examples(samples, sr)
