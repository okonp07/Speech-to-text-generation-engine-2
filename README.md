# Speech-to-Text Transcription App

[Open the live app](https://digit-classifier-model-1.streamlit.app)

This repository now powers a speech-to-text application that accepts microphone input or uploaded audio, transcribes spoken language into written text, and presents a confidence score alongside audio diagnostics.

Product requirements for this release: [`docs/speech-to-text-prd.md`](docs/speech-to-text-prd.md)

## Live app

- Streamlit app: [https://digit-classifier-model-1.streamlit.app](https://digit-classifier-model-1.streamlit.app)
- Repository: [https://github.com/okonp07/Speech-to-text-generation-engine](https://github.com/okonp07/Speech-to-text-generation-engine)

## What is in this repository

- `streamlit_app.py`: the interactive Streamlit app for microphone and file-based transcription.
- `digit_recognition/transcriber.py`: reusable speech-to-text module powered by `faster-whisper`.
- `digit_recognition/audio.py`: shared audio loading, normalization, waveform preparation, and quality checks.
- `tests/`: smoke tests for audio utilities, legacy digit-model code, and new transcription helpers.
- `Digit_Classification_from_Audio.ipynb`, `training.py`, and `evaluation.py`: legacy digit-classification artifacts retained for reference.

## Use the app

1. Open [https://digit-classifier-model-1.streamlit.app](https://digit-classifier-model-1.streamlit.app).
2. Choose a transcription model size from the sidebar.
3. Pick `Auto detect` or `English` as the language hint.
4. Record audio with your microphone or upload a supported audio file.
5. Review the transcript, confidence score, detected language, segment timeline, waveform, spectrogram, and audio-quality checks.

### Supported audio formats

- `WAV`
- `MP3`
- `M4A`
- `FLAC`
- `OGG`

### Best results

- Use clear speech and keep background noise low.
- Keep the microphone close enough for a clean recording.
- Speak naturally; the app is no longer limited to single digits.
- Expect the first run to take longer while the ASR model downloads.

## Run locally

```bash
git clone https://github.com/okonp07/Speech-to-text-generation-engine.git
cd Speech-to-text-generation-engine

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

If you use the project-local environment in this repo, prefer:

```powershell
.\SoT\Scripts\streamlit.exe run .\streamlit_app.py
```

Or activate the environment first so `streamlit run streamlit_app.py` resolves to the same interpreter:

```powershell
.\SoT\Scripts\Activate.ps1
streamlit run .\streamlit_app.py
```

Recommended Python version:

- Python `3.11` or `3.12`

If you deploy on Streamlit Community Cloud, choose the Python version from the deployment `Advanced settings`. Streamlit Community Cloud defaults can change over time, and the current dependency stack in this repo is most reliable on Python `3.11` or `3.12`. See the official docs: [Deploy your app](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) and [Manage dependencies](https://docs.streamlit.io/deploy/concepts/dependencies).

When Streamlit starts locally, open the URL shown in your terminal, usually:

```text
http://localhost:8501
```

## Python API

```python
from digit_recognition import SpeechTranscriber

transcriber = SpeechTranscriber(model_size="base")
result = transcriber.transcribe_file("sample.wav")

print(result.text)
print(result.confidence)
print(result.language)
```

You can also transcribe from a NumPy array:

```python
import librosa
from digit_recognition import SpeechTranscriber

audio, sr = librosa.load("sample.wav", sr=22050)
result = SpeechTranscriber(model_size="tiny").transcribe_array(audio, sample_rate=sr)

print(result.to_dict())
```

## Streamlit app

The app supports:

- live microphone recording in the browser
- single-file upload for WAV, MP3, M4A, FLAC, and OGG
- transcript output with confidence score
- detected language and language-confidence display
- segment-by-segment timing table
- waveform and mel-spectrogram visualizations
- lightweight audio-quality checks
- session transcription history

Run it locally:

```bash
streamlit run streamlit_app.py
```

## Project structure

```text
Speech-to-text-generation-engine/
├── .github/workflows/ci.yml
├── Digit_Classification_from_Audio.ipynb
├── Dockerfile
├── evaluation.py
├── models/
│   ├── enhanced_digit_model.pth
│   └── lightweight_digit_model.pth
├── digit_recognition/
│   ├── __init__.py
│   ├── audio.py
│   ├── datasets.py
│   ├── model.py
│   ├── predictor.py
│   ├── training.py
│   └── transcriber.py
├── requirements-dev.txt
├── requirements.txt
├── streamlit_app.py
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_audio.py
│   ├── test_models.py
│   └── test_transcriber.py
└── training.py
```

## Docker

```bash
docker build -t speech-transcription-app .
docker run -p 8501:8501 speech-transcription-app
```

## Development

```bash
python -m pip install -r requirements-dev.txt
pytest
flake8 digit_recognition tests training.py evaluation.py streamlit_app.py
```

## Notes

- `faster-whisper` downloads the selected transcription model on first use.
- Legacy digit-classification code remains in the repository for reference and backward compatibility.
