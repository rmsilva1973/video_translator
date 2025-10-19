# scripts/run_whisperx_align.py
import torch, json, os, whisperx

AUDIO = "work/audio/video_clean.wav"  # ou vocals
STT_JSON = "work/stt/video_stt.json"
MODEL_NAME = "large-v3"  # só para carregar metadata se quiser; alinhamento usa modelo acústico separado

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16

# 1) Carrega segmentos do Faster-Whisper
data = json.load(open(STT_JSON,"r",encoding="utf-8"))
segments = [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in data["segments"]]

# 2) Carrega modelo de alinhamento
model_a, metadata = whisperx.load_align_model(language_code="pt", device=device)

# 3) Realiza alinhamento
result_aligned = whisperx.align(segments, model_a, metadata, AUDIO, device, return_char_alignments=False)

# 4) Mescla de volta no JSON
for s, a in zip(data["segments"], result_aligned["segments"]):
    s["words"] = a.get("words", [])  # [{'word':'...', 'start':..., 'end':..., 'score':...}, ...]

os.makedirs("work/stt", exist_ok=True)
with open("work/stt/video_words_aligned.json","w",encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("OK: work/stt/video_words_aligned.json gerado")
