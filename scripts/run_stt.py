import json, os
from pathlib import Path

class SpeechToText:
    def __init__(self, video_name):
        self.video_name = video_name
        self.audio_path = f"work/audio/{Path(video_name).stem}_clean.wav"
        self.output_path = f"work/stt/{Path(video_name).stem}_stt.json"
        self.model_name = "large-v3"
        self.device = "cuda"
        self.compute_type = "float16" if self.device=="cuda" else "int8"

    def process(self):
        from faster_whisper import WhisperModel
        
        model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type)

        segments, info = model.transcribe(
          self.audio_path,
          language="pt",
          vad_filter=True,
          beam_size=5,
          temperature=0.0,
          word_timestamps=False,
        )

        out_segments = []
        for seg in segments:
          out_segments.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
            "avg_logprob": seg.avg_logprob,
            "no_speech_prob": seg.no_speech_prob,
            "words": []
          })

        os.makedirs("work/stt", exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
          json.dump({
            "language": info.language,
            "duration": info.duration,
            "model": self.model_name,
            "segments": out_segments
          }, f, ensure_ascii=False, indent=2)
        print(f"OK: {self.output_path} gerado")
