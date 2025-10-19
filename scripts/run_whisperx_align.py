import json, os
from pathlib import Path

class WhisperXAlign:
    def __init__(self, video_name):
        self.video_name = video_name
        self.audio_path = f"work/audio/{Path(video_name).stem}_clean.wav"
        self.stt_json_path = f"work/stt/{Path(video_name).stem}_stt.json"
        self.output_path = f"work/stt/{Path(video_name).stem}_words_aligned.json"
        self.batch_size = 16

    def process(self):
        import torch, whisperx
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 1) Carrega segmentos do Faster-Whisper
        data = json.load(open(self.stt_json_path,"r",encoding="utf-8"))
        segments = [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in data["segments"]]

        # 2) Carrega modelo de alinhamento
        model_a, metadata = whisperx.load_align_model(language_code="pt", device=self.device)

        # 3) Realiza alinhamento
        result_aligned = whisperx.align(segments, model_a, metadata, self.audio_path, self.device, return_char_alignments=False)

        # 4) Mescla de volta no JSON
        for s, a in zip(data["segments"], result_aligned["segments"]):
            s["words"] = a.get("words", [])

        os.makedirs("work/stt", exist_ok=True)
        with open(self.output_path,"w",encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"OK: {self.output_path} gerado")
