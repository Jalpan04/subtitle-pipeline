# Subtitle Pipeline

![GitHub top language](https://img.shields.io/github/languages/top/Jalpan04/subtitle-pipeline) ![GitHub repo size](https://img.shields.io/github/repo-size/Jalpan04/subtitle-pipeline) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

A robust pipeline for generating Romanised Hindi (Hinglish) subtitles from video and audio files.

## Features

- **Accurate Transcription**: Uses OpenAI's Whisper (turbo model) for high-quality native Devanagari transcription.
- **Dynamic Subtitles**: Word-level timestamps provide precise, short subtitle segments.
- **Hinglish Transliteration**: Custom rule-based transliterator converts Devanagari to Roman script.
- **ML-Powered Correction**: A Character N-Gram Machine Learning model classifies and corrects English loanwords while preserving Hindi.

## Installation

```bash
pip install openai-whisper autocorrect scikit-learn
```

## Usage

```bash
python pipeline_transcriber.py "your_video.mp4"
```

The script will generate a `.srt` file in the same directory.

## License

This project is licensed under the Apache-2.0 License - see the [LICENSE](LICENSE) file for details.
