# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Defines routines to compute mel spectrogram features from audio waveform."""
import time

import librosa
import numpy as np
from pydub import AudioSegment

from vggish import vggish_params
import resampy
import soundfile as sf

from logger import get_logger

logger = get_logger('mel')


def frame(data, window_length, hop_length):
    """Convert array into a sequence of successive possibly overlapping frames.

    An n-dimensional array of shape (num_samples, ...) is converted into an
    (n+1)-D array of shape (num_frames, window_length, ...), where each frame
    starts hop_length points after the preceding one.

    This is accomplished using stride_tricks, so the original data is not
    copied.  However, there is no zero-padding, so any incomplete frames at the
    end are not included.

    Args:
      data: np.array of dimension N >= 1.
      window_length: Number of samples in each frame.
      hop_length: Advance (in samples) between each window.

    Returns:
      (N+1)-D np.array with as many rows as there are complete frames that can be
      extracted.
    """
    num_samples = data.shape[0]
    num_frames = 1 + int(np.floor((num_samples - window_length) / hop_length))
    shape = (num_frames, window_length) + data.shape[1:]
    strides = (data.strides[0] * hop_length,) + data.strides
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


def periodic_hann(window_length):
    """Calculate a "periodic" Hann window.

    The classic Hann window is defined as a raised cosine that starts and
    ends on zero, and where every value appears twice, except the middle
    point for an odd-length window.  Matlab calls this a "symmetric" window
    and np.hanning() returns it.  However, for Fourier analysis, this
    actually represents just over one cycle of a period N-1 cosine, and
    thus is not compactly expressed on a length-N Fourier basis.  Instead,
    it's better to use a raised cosine that ends just before the final
    zero value - i.e. a complete cycle of a period-N cosine.  Matlab
    calls this a "periodic" window. This routine calculates it.

    Args:
      window_length: The number of points in the returned window.

    Returns:
      A 1D np.array containing the periodic hann window.
    """
    return 0.5 - (0.5 * np.cos(2 * np.pi / window_length *
                               np.arange(window_length)))


def stft_magnitude(signal, fft_length,
                   hop_length=None,
                   window_length=None):
    """Calculate the short-time Fourier transform magnitude.

    Args:
      signal: 1D np.array of the input time-domain signal.
      fft_length: Size of the FFT to apply.
      hop_length: Advance (in samples) between each frame passed to FFT.
      window_length: Length of each block of samples to pass to FFT.

    Returns:
      2D np.array where each row contains the magnitudes of the fft_length/2+1
      unique values of the FFT for the corresponding frame of input samples.
    """
    frames = frame(signal, window_length, hop_length)
    # Apply frame window to each frame. We use a periodic Hann (cosine of period
    # window_length) instead of the symmetric Hann of np.hanning (period
    # window_length-1).
    window = periodic_hann(window_length)
    windowed_frames = frames * window
    return np.abs(np.fft.rfft(windowed_frames, int(fft_length)))


# Mel spectrum constants and functions.
_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0


def hertz_to_mel(frequencies_hertz):
    """Convert frequencies to mel scale using HTK formula.

    Args:
      frequencies_hertz: Scalar or np.array of frequencies in hertz.

    Returns:
      Object of same size as frequencies_hertz containing corresponding values
      on the mel scale.
    """
    return _MEL_HIGH_FREQUENCY_Q * np.log(
        1.0 + (frequencies_hertz / _MEL_BREAK_FREQUENCY_HERTZ))


def spectrogram_to_mel_matrix(num_mel_bins=20,
                              num_spectrogram_bins=129,
                              audio_sample_rate=8000,
                              lower_edge_hertz=125.0,
                              upper_edge_hertz=3800.0):
    """Return a matrix that can post-multiply spectrogram rows to make mel.

    Returns a np.array matrix A that can be used to post-multiply a matrix S of
    spectrogram values (STFT magnitudes) arranged as frames x bins to generate a
    "mel spectrogram" M of frames x num_mel_bins.  M = S A.

    The classic HTK algorithm exploits the complementarity of adjacent mel bands
    to multiply each FFT bin by only one mel weight, then add it, with positive
    and negative signs, to the two adjacent mel bands to which that bin
    contributes.  Here, by expressing this operation as a matrix multiply, we go
    from num_fft multiplies per frame (plus around 2*num_fft adds) to around
    num_fft^2 multiplies and adds.  However, because these are all presumably
    accomplished in a single call to np.dot(), it's not clear which approach is
    faster in Python.  The matrix multiplication has the attraction of being more
    general and flexible, and much easier to read.

    Args:
      num_mel_bins: How many bands in the resulting mel spectrum.  This is
        the number of columns in the output matrix.
      num_spectrogram_bins: How many bins there are in the source spectrogram
        data, which is understood to be fft_size/2 + 1, i.e. the spectrogram
        only contains the nonredundant FFT bins.
      audio_sample_rate: Samples per second of the audio at the input to the
        spectrogram. We need this to figure out the actual frequencies for
        each spectrogram bin, which dictates how they are mapped into mel.
      lower_edge_hertz: Lower bound on the frequencies to be included in the mel
        spectrum.  This corresponds to the lower edge of the lowest triangular
        band.
      upper_edge_hertz: The desired top edge of the highest frequency band.

    Returns:
      An np.array with shape (num_spectrogram_bins, num_mel_bins).

    Raises:
      ValueError: if frequency edges are incorrectly ordered or out of range.
    """
    nyquist_hertz = audio_sample_rate / 2.
    if lower_edge_hertz < 0.0:
        raise ValueError("lower_edge_hertz %.1f must be >= 0" % lower_edge_hertz)
    if lower_edge_hertz >= upper_edge_hertz:
        raise ValueError("lower_edge_hertz %.1f >= upper_edge_hertz %.1f" %
                         (lower_edge_hertz, upper_edge_hertz))
    if upper_edge_hertz > nyquist_hertz:
        raise ValueError("upper_edge_hertz %.1f is greater than Nyquist %.1f" %
                         (upper_edge_hertz, nyquist_hertz))
    spectrogram_bins_hertz = np.linspace(0.0, nyquist_hertz, num_spectrogram_bins)
    spectrogram_bins_mel = hertz_to_mel(spectrogram_bins_hertz)
    # The i'th mel band (starting from i=1) has center frequency
    # band_edges_mel[i], lower edge band_edges_mel[i-1], and higher edge
    # band_edges_mel[i+1].  Thus, we need num_mel_bins + 2 values in
    # the band_edges_mel arrays.
    band_edges_mel = np.linspace(hertz_to_mel(lower_edge_hertz),
                                 hertz_to_mel(upper_edge_hertz), num_mel_bins + 2)
    # Matrix to post-multiply feature arrays whose rows are num_spectrogram_bins
    # of spectrogram values.
    mel_weights_matrix = np.empty((num_spectrogram_bins, num_mel_bins))
    for i in range(num_mel_bins):
        lower_edge_mel, center_mel, upper_edge_mel = band_edges_mel[i:i + 3]
        # Calculate lower and upper slopes for every spectrogram bin.
        # Line segments are linear in the *mel* domain, not hertz.
        lower_slope = ((spectrogram_bins_mel - lower_edge_mel) /
                       (center_mel - lower_edge_mel))
        upper_slope = ((upper_edge_mel - spectrogram_bins_mel) /
                       (upper_edge_mel - center_mel))
        # .. then intersect them with each other and zero.
        mel_weights_matrix[:, i] = np.maximum(0.0, np.minimum(lower_slope,
                                                              upper_slope))
    # HTK excludes the spectrogram DC bin; make sure it always gets a zero
    # coefficient.
    mel_weights_matrix[0, :] = 0.0
    return mel_weights_matrix


def log_mel_spectrogram(data,
                        audio_sample_rate=8000,
                        log_offset=0.0,
                        window_length_secs=0.025,
                        hop_length_secs=0.010,
                        **kwargs):
    """Convert waveform to a log magnitude mel-frequency spectrogram.

    Args:
      data: 1D np.array of waveform data.
      audio_sample_rate: The sampling rate of data.
      log_offset: Add this to values when taking log to avoid -Infs.
      window_length_secs: Duration of each window to analyze.
      hop_length_secs: Advance between successive analysis windows.
      **kwargs: Additional arguments to pass to spectrogram_to_mel_matrix.

    Returns:
      2D np.array of (num_frames, num_mel_bins) consisting of log mel filterbank
      magnitudes for successive frames.
    """
    window_length_samples = int(round(audio_sample_rate * window_length_secs))
    hop_length_samples = int(round(audio_sample_rate * hop_length_secs))
    fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))
    spectrogram = stft_magnitude(
        data,
        fft_length=fft_length,
        hop_length=hop_length_samples,
        window_length=window_length_samples)
    mel_spectrogram = np.dot(spectrogram, spectrogram_to_mel_matrix(
        num_spectrogram_bins=spectrogram.shape[1],
        audio_sample_rate=audio_sample_rate, **kwargs))
    return np.log(mel_spectrogram + log_offset)


def waveform_to_examples(data, sample_rate, vgg_params):
    """Converts audio waveform into an array of examples for VGGish.
    Args:
      data: np.array of either one dimension (mono) or two dimensions
        (multi-channel, with the outer dimension representing channels).
        Each sample is generally expected to lie in the range [-1.0, +1.0],
        although this is not required.
      sample_rate: Sample rate of data.
      vgg_params: vgg_params
    Returns:
      3-D np.array of shape [num_examples, num_frames, num_bands] which represents
      a sequence of examples, each of which contains a patch of log mel
      spectrogram, covering num_frames frames of audio and num_bands mel frequency
      bands, where the frame length is vggish_params.STFT_HOP_LENGTH_SECONDS.
    """
    # Convert to mono.
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    # Resample to the rate assumed by VGGish.
    if sample_rate != vgg_params.SAMPLE_RATE:
        t = time.time()
        data = resampy.resample(data, sample_rate, vgg_params.SAMPLE_RATE, filter='kaiser_fast')
        logger.debug('resample took %s' % (time.time() - t))

    # Compute log mel spectrogram features.
    t = time.time()
    log_mel = log_mel_spectrogram(
        data,
        audio_sample_rate=vgg_params.SAMPLE_RATE,
        log_offset=vgg_params.LOG_OFFSET,
        window_length_secs=vgg_params.STFT_WINDOW_LENGTH_SECONDS,
        hop_length_secs=vgg_params.STFT_HOP_LENGTH_SECONDS,
        num_mel_bins=vgg_params.NUM_MEL_BINS,
        lower_edge_hertz=vgg_params.MEL_MIN_HZ,
        upper_edge_hertz=vgg_params.MEL_MAX_HZ)
    logger.debug('log_mel_spectrogram took %s' % (time.time() - t))

    if vgg_params.EXAMPLE_WINDOW_SECONDS is not None:
        t = time.time()
        # Frame features into examples.
        features_sample_rate = 1.0 / vgg_params.STFT_HOP_LENGTH_SECONDS
        example_window_length = int(round(
            vgg_params.EXAMPLE_WINDOW_SECONDS * features_sample_rate))
        example_hop_length = int(round(
            vgg_params.EXAMPLE_HOP_SECONDS * features_sample_rate))
        log_mel_examples = frame(
            log_mel,
            window_length=example_window_length,
            hop_length=example_hop_length)
        logger.debug('framing took %s' % (time.time() - t))
        return log_mel_examples
    else:
        return log_mel


def wavfile_to_examples(wav_file, vgg_params):
    """Convenience wrapper around waveform_to_examples() for a common WAV format.
    Args:
      wav_file: String path to a file, or a file-like object. The file
      is assumed to contain WAV audio data with signed 16-bit PCM samples.
      vgg_params: vgg_params
    Returns:
      See waveform_to_examples.
    """

    t = time.time()
    logger.debug('proc %s' % wav_file)
    wav_data, sr = librosa.core.load(wav_file,
                                     sr=None)  # sf.read(wav_file, dtype=np.int16)#librosa.core.load(wav_file, dtype=np.int16, sr=None)
    logger.debug('read took %s' % (time.time() - t))
    samples = wav_data
    assert -1.00002 < np.max(wav_data) < 1.00002
    assert -1.00002 < np.min(wav_data) < 1.00002
    assert np.abs(np.max(wav_data) - np.min(wav_data)) > 0.001
    # assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
    # samples = wav_data / 32768.0  # Convert to [-1.0, +1.0]
    return waveform_to_examples(samples, sr, vgg_params)


def wavfile_to_signal(wav_file, vgg_params):
    t = time.time()
    wav_data, sr = librosa.core.load(wav_file,
                                     sr=None)  # sf.read(wav_file, dtype=np.int16)#librosa.core.load(wav_file, dtype=np.int16, sr=None)
    logger.debug('read took %s' % (time.time() - t))
    samples = wav_data
    assert -1.00002 < np.max(wav_data) < 1.00002
    assert -1.00002 < np.min(wav_data) < 1.00002
    # assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
    # samples = wav_data / 32768.0  # Convert to [-1.0, +1.0]
    if len(samples.shape) > 1:
        data = np.mean(samples, axis=1)
    # Resample to the rate assumed by VGGish.
    if sr != vgg_params.SAMPLE_RATE:
        t = time.time()
        samples = resampy.resample(samples, sr, vgg_params.SAMPLE_RATE, filter='kaiser_fast')
        logger.debug('resample took %s' % (time.time() - t))

    return samples
