import json, os, numpy as np
from pathlib import Path
import librosa, pyworld as pw

class ProsodySSMLGenerator:
    def __init__(self, video_name):
        self.video_name = video_name
        self.audio = Path(f"work/audio/{Path(video_name).stem}_clean.wav")
        self.words_json = Path(f"work/stt/{Path(video_name).stem}_words_aligned.json")
        self.mt_json = Path(f"work/mt/{Path(video_name).stem}_en_segments.json")
        self.out_json = Path(f"work/ssml/{Path(video_name).stem}_en_ssml.json")
        self.out_srt = Path(f"work/ssml/{Path(video_name).stem}_en_ssml_preview.srt")
        self.out_log = Path(f"logs/{Path(video_name).stem}_ssml_report.json")
        
        os.makedirs(self.out_json.parent, exist_ok=True)
        os.makedirs(self.out_log.parent, exist_ok=True)
        
        self.y = None
        self.sr = 16000
        self.frame_len = 0.02
        self.hop_len = None

    def _segment_audio(self, start, end):
        s = max(0, int(start * self.sr))
        e = min(len(self.y), int(end * self.sr))
        return self.y[s:e]

    def _compute_rms(self, sig):
        if len(sig) == 0:
            return 0.0
        return float(np.sqrt(np.mean(sig**2) + 1e-12))

    def _estimate_wps(self, words, start, end):
        dur = max(0.01, end - start)
        wc = max(1, len(words))
        return wc / dur

    def _classify_rate(self, wps, ref_wps=(3.0, 3.5)):
        low, high = ref_wps
        if wps < low:
            delta = min(12, int((low - wps) * 6))
            return +delta
        elif wps > high:
            delta = min(12, int((wps - high) * 6))
            return -delta
        return 0

    def _compute_f0(self, sig):
        if len(sig) < self.sr * 0.15:
            return None
        _f0, t = pw.harvest(sig.astype(np.float64), self.sr, f0_floor=50.0, f0_ceil=300.0)
        f0 = pw.stonemask(sig.astype(np.float64), _f0, t, self.sr)
        f0 = f0[f0 > 1.0]
        if len(f0) == 0:
            return None
        return f0

    def _classify_pitch_trend(self, sig):
        f0 = self._compute_f0(sig)
        if f0 is None or len(f0) < 5:
            return "neutral"
        x = np.arange(len(f0))
        x1 = int(len(f0) * 0.3)
        x2 = int(len(f0) * 0.9)
        if x2 <= x1:
            return "neutral"
        coeff = np.polyfit(x[x1:x2], f0[x1:x2], 1)
        slope = coeff[0]
        if slope > 0.1:
            return "question"
        if slope < -0.1:
            return "statement"
        if np.max(f0) > (np.median(f0) * 1.15):
            return "emphasis"
        return "neutral"

    def _quantize_pause(self, delta_s):
        if delta_s < 0.15:
            return None
        if delta_s < 0.30:
            return 0.20
        if delta_s < 0.60:
            return 0.40
        if delta_s < 1.00:
            return 0.70
        return min(1.20, round(delta_s, 2))

    def _to_srt_time(self, t):
        h = int(t//3600); m = int((t%3600)//60); s = t%60
        return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")

    def _ptidx_to_enidx(self, pt_idx, words_len, en_tokens_len):
        if words_len <= 1:
            return en_tokens_len
        return int(round((pt_idx / (words_len - 1)) * max(0, en_tokens_len - 1)))

    def process(self):
        aligned = json.load(open(self.words_json, "r", encoding="utf-8"))
        en_map = json.load(open(self.mt_json, "r", encoding="utf-8"))
        pt_segments = aligned["segments"]
        en_segments = en_map["segments"]
        
        assert len(pt_segments) == len(en_segments), "PT and EN segmentation length mismatch"
        
        self.y, self.sr = librosa.load(str(self.audio), sr=16000, mono=True)
        self.hop_len = int(self.sr * self.frame_len)
        
        out_segments = []
        report = []
        
        for i, (pt, en) in enumerate(zip(pt_segments, en_segments), start=1):
            start, end = float(pt["start"]), float(pt["end"])
            words = pt.get("words") or []
            en_text = en["en_text"]
            dur = max(0.01, end - start)
            
            pauses = []
            for j in range(1, len(words)):
                prev_end = float(words[j-1]["end"])
                cur_start = float(words[j]["start"])
                gap = cur_start - prev_end
                q = self._quantize_pause(gap)
                if q:
                    pauses.append({"after_index": j-1, "dur": q})
            
            wps = self._estimate_wps(words, start, end)
            rate_pct = self._classify_rate(wps)
            
            sig = self._segment_audio(start, end)
            pitch_cat = self._classify_pitch_trend(sig)
            
            if pitch_cat == "question":
                pitch_ssml = "+2st"
            elif pitch_cat == "emphasis":
                pitch_ssml = "+1st"
            elif pitch_cat == "statement":
                pitch_ssml = "-1st"
            else:
                pitch_ssml = "0st"
            
            en_tokens = en_text.split()
            ssml_parts = []
            token_cursor = 0
            
            for p in pauses:
                en_idx = self._ptidx_to_enidx(p["after_index"], len(words), len(en_tokens))
                ssml_parts.append(" ".join(en_tokens[token_cursor:en_idx+1]))
                ssml_parts.append(f'<break time="{int(p["dur"]*1000)}ms"/>')
                token_cursor = en_idx + 1
            
            if token_cursor < len(en_tokens):
                ssml_parts.append(" ".join(en_tokens[token_cursor:]))
            
            base_text_with_breaks = " ".join([s for s in ssml_parts if s])
            rate_str = f"{rate_pct}%" if rate_pct >= 0 else f"{rate_pct}%"
            ssml = f'<prosody rate="{rate_str}" pitch="{pitch_ssml}">{base_text_with_breaks}</prosody>'
            
            out_segments.append({
                "start": start, "end": end, "en_text": en_text, "ssml": ssml,
                "wps_pt": wps, "rate_pct": rate_pct, "pitch_cat": pitch_cat,
                "pauses_count": len(pauses)
            })
            
            report.append({
                "idx": i, "start": start, "end": end, "dur": dur,
                "wps_pt": round(wps, 2), "rate_pct": rate_pct,
                "pitch_cat": pitch_cat, "pauses": pauses[:8]
            })
        
        json.dump({"segments": out_segments}, open(self.out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        json.dump(report, open(self.out_log, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        
        with open(self.out_srt, "w", encoding="utf-8") as f:
            for idx, s in enumerate(out_segments, 1):
                preview = s["ssml"].replace("<prosody", "[prosody").replace("</prosody>", "[/prosody]").replace('<break time="', "[pause:").replace('"/>', "ms]")
                f.write(f"{idx}\n{self._to_srt_time(s['start'])} --> {self._to_srt_time(s['end'])}\n{preview}\n\n")
        
        print(f"OK: {self.out_json}, {self.out_srt} | Report: {self.out_log}")
