# Audio Denoising using FFT

This script performs basic **audio signal enhancement** by removing specific noise frequencies from a WAV file using the **Fast Fourier Transform (FFT)**.

## ✨ Features

- Load an audio file (`.wav`)
- Transform signal to frequency domain using FFT
- Zero out specified noise frequencies
- Reconstruct the cleaned signal via inverse FFT
- Save the result as a new audio file

## 🧪 Example

```python
input_audio_file = "q1.wav"
output_audio_file = "q1_output.wav"
noise_frequencies = [500, 1000]  # in Hz
```

## 📚 Related Topics

- Audio processing
- Frequency domain filtering
- FFT and IFFT

## ▶️ Run

```bash
python ex2.py
```

Make sure the input `.wav` file exists in the working directory.
