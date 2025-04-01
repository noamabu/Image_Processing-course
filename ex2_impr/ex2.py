import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib

def clean_audio(input_file, output_file, noise_frequencies):
    # Load audio file
    sample_rate, audio_data = wavfile.read(input_file)

    # Perform FFT on audio data
    audio_fft = np.fft.fft(audio_data)

    # Identify and filter out noise frequencies
    for freq in noise_frequencies:
        bin_index = int(freq * len(audio_data) / sample_rate)
        audio_fft[bin_index] = 0  # Set the corresponding frequency component to zero

    # Perform Inverse FFT to get cleaned audio data
    cleaned_audio_data = np.fft.ifft(audio_fft).real.astype(np.int16)

    # Save the cleaned audio to a new file
    wavfile.write(output_file, sample_rate, cleaned_audio_data)

# Example usage:
input_audio_file = "q1.wav"
output_audio_file = "q1_output.wav"
noise_frequencies = [500, 1000]  # Example noise frequencies in Hz

clean_audio(input_audio_file, output_audio_file, noise_frequencies)
