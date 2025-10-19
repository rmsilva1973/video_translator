# scripts/run_stt.py
from faster_whisper import WhisperModel
import json, os

audio_path = "work/audio/video_clean.wav"  # ou aula_vocals_16k.wav se preferiu
model_name = "large-v3"               # ou "large-v3"
device = "cuda"                     # "cpu" se não tiver GPU
compute_type = "float16" if device=="cuda" else "int8"

model = WhisperModel(model_name, device=device, compute_type=compute_type)

segments, info = model.transcribe(
  audio_path,
  language="pt",          # força PT-BR
  vad_filter=True,        # ajuda em silêncios
  beam_size=5,            # 5-10 melhora estabilidade
  temperature=0.0,        # conservador para reduzir variação
  word_timestamps=False,  # usaremos WhisperX para alinhamento fino
)

out_segments = []
for seg in segments:
  out_segments.append({
    "start": seg.start,
    "end": seg.end,
    "text": seg.text.strip(),
    "avg_logprob": seg.avg_logprob,
    "no_speech_prob": seg.no_speech_prob,
    "words": []  # preencheremos depois com WhisperX
  })

os.makedirs("work/stt", exist_ok=True)
with open("work/stt/video_stt.json", "w", encoding="utf-8") as f:
  json.dump({
    "language": info.language,
    "duration": info.duration,
    "model": model_name,
    "segments": out_segments
  }, f, ensure_ascii=False, indent=2)
print("OK: work/stt/video_stt.json gerado")
