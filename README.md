# 🎬 Video Translation Pipeline

A comprehensive AI-powered pipeline for translating Brazilian Portuguese videos to English with natural prosody preservation.

## 🌟 Features

- 🎵 **Audio Extraction & Enhancement** - Extracts and cleans audio with noise reduction
- 🗣️ **Advanced Speech Recognition** - Uses Faster-Whisper large-v3 for accurate transcription
- 🎯 **Word-Level Alignment** - Precise timing with WhisperX alignment
- 📝 **Text Post-Processing** - Punctuation restoration and normalization
- 🌐 **Machine Translation** - PT→EN translation with quality optimization
- 🎭 **Prosody Analysis** - Preserves speech patterns and generates SSML markup
- 🚀 **Batch Processing** - Processes multiple videos automatically
- ⚡ **Fast Startup** - Lazy loading for quick application launch

## 🏗️ Architecture

The pipeline follows a 6-step process:

```
📹 Input Video → 🎵 Audio → 🗣️ STT → 🎯 Align → 📝 Process → 🌐 Translate → 🎭 SSML
```

### Pipeline Steps

1. **🎵 Audio Extraction** (`extract_audio.py`)
   - Converts video to 16kHz mono WAV
   - Applies audio filters (loudnorm, highpass, lowpass)
   - Optional noise reduction with RNNoise

2. **🗣️ Speech-to-Text** (`run_stt.py`)
   - Uses Faster-Whisper large-v3 model
   - Portuguese language optimization
   - High-quality transcription with confidence scores

3. **🎯 Word Alignment** (`run_whisperx_align.py`)
   - WhisperX for precise word-level timestamps
   - Essential for prosody analysis
   - Handles alignment failures gracefully

4. **📝 Portuguese Post-Processing** (`pt_postprocess.py`)
   - Punctuation restoration with DeepMultilingualPunctuation
   - Text normalization (numbers, units, IPs)
   - Glossary support for domain-specific terms
   - Optional LanguageTool grammar correction

5. **🌐 Machine Translation** (`mt_translate.py`)
   - Multiple MT models (NLLB, M2M100, Opus-MT)
   - Quality-based model selection
   - Language detection and validation
   - Glossary integration

6. **🎭 Prosody & SSML** (`prosody_and_ssml.py`)
   - Speech rate analysis from original audio
   - Pitch pattern detection
   - Pause insertion based on word timing
   - SSML generation for natural TTS

## 🚀 Quick Start

### Prerequisites

- Python 3.13+
- CUDA-compatible GPU (recommended)
- FFmpeg installed on system

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd translate_videos
   ```

2. **Install dependencies**
   ```bash
   uv sync
   ```

3. **Prepare input**
   ```bash
   # Place your video files in the input/ directory
   cp your_video.mp4 input/
   ```

4. **Run the pipeline**
   ```bash
   uv run main.py
   ```

## 📁 Project Structure

```
translate_videos/
├── 📄 main.py                 # Main application entry point
├── 📁 scripts/               # Pipeline modules
│   ├── extract_audio.py      # Audio extraction & cleaning
│   ├── run_stt.py            # Speech-to-text processing
│   ├── run_whisperx_align.py # Word alignment
│   ├── pt_postprocess.py     # Portuguese text processing
│   ├── mt_translate.py       # Machine translation
│   └── prosody_and_ssml.py   # Prosody analysis & SSML
├── 📁 input/                 # Input videos (.gitkeep)
├── 📁 work/                  # Temporary processing files
│   ├── audio/               # Extracted audio files
│   ├── stt/                 # Speech recognition results
│   ├── mt/                  # Translation outputs
│   └── ssml/                # Final SSML results
├── 📁 logs/                  # Processing logs and reports
├── 📁 output/                # Final output files (.gitkeep)
├── 📁 glossary/              # Domain-specific terminology
│   └── terms.csv            # Glossary terms (find,replace,flags)
├── 📁 models/                # ML model cache (.gitkeep)
└── 📄 README.md              # This file
```

## 🎯 Usage Examples

### Basic Usage
```bash
# Process all videos in input/ folder
uv run main.py
```

### Supported Video Formats
- 📹 MP4, AVI, MOV, MKV
- 🎬 WMV, FLV, WebM

### Output Files

For each input video `video.mp4`, the pipeline generates:

**Audio Processing:**
- `work/audio/video_16k_mono.wav` - Processed audio
- `work/audio/video_clean.wav` - Noise-reduced audio

**Speech Recognition:**
- `work/stt/video_stt.json` - Raw transcription
- `work/stt/video_words_aligned.json` - Word-aligned transcription
- `work/stt/video_pt_clean.json` - Post-processed Portuguese
- `work/stt/video_pt_clean.srt` - Portuguese subtitles

**Translation:**
- `work/mt/video_en_segments.json` - English translation
- `work/mt/video_en.srt` - English subtitles
- `logs/video_mt_report.json` - Translation quality report

**SSML Generation:**
- `work/ssml/video_en_ssml.json` - SSML with prosody
- `work/ssml/video_en_ssml_preview.srt` - SSML preview
- `logs/video_ssml_report.json` - Prosody analysis report

## ⚙️ Configuration

### Glossary Setup

Create `glossary/terms.csv` for domain-specific translations:
```csv
termo técnico,technical term,i
API,API,
servidor,server,i
```

Format: `find,replace,flags` (flags: `i` for case-insensitive)

### Model Configuration

The pipeline uses these models by default:
- **STT**: `faster-whisper/large-v3`
- **Alignment**: WhisperX Portuguese model
- **Translation**: NLLB-200-1.3B (with fallbacks)
- **Punctuation**: DeepMultilingualPunctuation

## 🔧 Advanced Features

### Audio Enhancement Options

The pipeline supports multiple audio cleaning methods:
- **Basic**: Loudnorm + frequency filtering
- **Advanced**: Spectral noise reduction
- **RNNoise**: Deep learning-based denoising (requires model)

### Translation Quality Control

- **Multi-model approach**: Primary + fallback models
- **Language detection**: Validates translation quality
- **Length ratio analysis**: Detects translation issues
- **Confidence scoring**: Quality metrics for each segment

### Prosody Preservation

- **Speech rate analysis**: Maintains original pacing
- **Pitch pattern detection**: Preserves intonation
- **Pause insertion**: Natural speech rhythm
- **SSML generation**: Compatible with modern TTS systems

## 🐛 Troubleshooting

### Common Issues

**🚨 CUDA Out of Memory**
```bash
# Reduce batch size or use CPU
export CUDA_VISIBLE_DEVICES=""
```

**🚨 Model Download Fails**
```bash
# Check internet connection and Hugging Face access
huggingface-cli login
```

**🚨 Audio Processing Errors**
```bash
# Verify FFmpeg installation
ffmpeg -version
```

**🚨 Alignment Warnings**
- Normal for difficult audio segments
- Pipeline continues with fallback timing
- Check audio quality if excessive

### Performance Optimization

**GPU Usage:**
- Automatic CUDA detection
- Mixed precision for memory efficiency
- Batch processing optimization

**Memory Management:**
- Lazy model loading
- Automatic cleanup between steps
- Configurable batch sizes

## 📊 Performance Metrics

Typical processing times (3-minute video, RTX 4090):
- 🎵 Audio extraction: ~5 seconds
- 🗣️ Speech-to-text: ~30 seconds
- 🎯 Word alignment: ~45 seconds
- 📝 Post-processing: ~2 seconds
- 🌐 Translation: ~10 seconds
- 🎭 SSML generation: ~5 seconds

**Total**: ~1.5 minutes for 3-minute video

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Faster-Whisper** - High-performance speech recognition
- **WhisperX** - Precise word-level alignment
- **NLLB** - State-of-the-art neural machine translation
- **DeepMultilingualPunctuation** - Punctuation restoration
- **PyWorld** - Prosody analysis tools

## 📞 Support

For issues and questions:
- 🐛 **Bug Reports**: Open an issue with detailed reproduction steps
- 💡 **Feature Requests**: Describe your use case and requirements
- 📖 **Documentation**: Check this README and inline code comments
- 🔧 **Technical Support**: Include system specs and error logs

---

**Made with ❤️ for the Brazilian Portuguese community**
