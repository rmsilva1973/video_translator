import json, re, csv, yaml, os, sys
from pathlib import Path

class PortuguesePostProcessor:
    def __init__(self, video_name):
        self.video_name = video_name
        self.in_json = Path(f"work/stt/{Path(video_name).stem}_words_aligned.json")
        self.out_json = Path(f"work/stt/{Path(video_name).stem}_pt_clean.json")
        self.out_srt = Path(f"work/stt/{Path(video_name).stem}_pt_clean.srt")
        self.glossary_csv = Path("glossary/terms.csv")
        self.itn_yaml = Path("glossary/itn_rules.yaml")
        
        # Initialize punctuation model (lazy loading)
        self.USE_PUNCTUATOR = True
        self.punct_model = None

        # Initialize LanguageTool
        self.USE_LT = False
        self.LT_SERVER = "http://localhost:8081"
        try:
            import language_tool_python
            self.tool = language_tool_python.LanguageToolPublicAPI("pt-BR", self.LT_SERVER)
            self.USE_LT = True
        except Exception:
            print("[INFO] LanguageTool não está ativo; seguindo sem LT local.")

    def process(self):
        subs = self.read_glossary(self.glossary_csv)
        data = json.load(open(self.in_json, "r", encoding="utf-8"))
        cleaned_segments = []

        for seg in data["segments"]:
            raw = seg["text"]
            txt = self.restore_punctuation(raw)
            if not self.USE_PUNCTUATOR:
                txt = self.basic_truecase(txt)

            txt = self.normalize_ips(txt)
            txt = self.normalize_numbers_units(txt)
            txt = self.apply_glossary(txt, subs)
            txt = self.lt_fix(txt)

            new_seg = {
                "start": seg["start"],
                "end": seg["end"],
                "text": txt.strip(),
                "words": seg.get("words", [])
            }
            cleaned_segments.append(new_seg)

        os.makedirs("work/stt", exist_ok=True)
        json.dump({"segments": cleaned_segments}, open(self.out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

        lines = []
        for seg in cleaned_segments:
            words = seg.get("words") or []
            if words:
                lines.extend(self.split_for_srt(words))
            else:
                lines.append({"start": seg["start"], "end": seg["end"], "text": seg["text"]})

        self.write_srt(lines, self.out_srt)
        print("OK:", self.out_json, self.out_srt)

    def read_glossary(self, glossary_csv):
        subs = []
        if Path(glossary_csv).exists():
            with open(glossary_csv, newline="", encoding="utf-8") as f:
                r = csv.DictReader(f, fieldnames=["find","replace","flags"])
                for row in r:
                    find = row["find"].strip()
                    repl = row["replace"].strip()
                    flags = (row.get("flags") or "").strip().lower()
                    re_flags = re.IGNORECASE if "i" in flags else 0
                    subs.append((re.compile(re.escape(find), re_flags), repl))
        return subs

    def normalize_numbers_units(self, text):
        text = re.sub(r"(?i)\b(\d+)(\s*)(ghz)\b", r"\1,0 GHz", text)
        text = text.replace(" gigahertz", " GHz")
        text = text.replace(" megahertz", " MHz")
        text = text.replace(" gigabytes", " GB").replace(" gigabyte", " GB")
        text = text.replace(" megabytes", " MB").replace(" megabyte", " MB")
        return text

    def normalize_ips(self, text):
        return text

    def restore_punctuation(self, text):
        if self.USE_PUNCTUATOR:
            try:
                if self.punct_model is None:
                    from deepmultilingualpunctuation import PunctuationModel
                    self.punct_model = PunctuationModel()
                return self.punct_model.restore_punctuation(text)
            except Exception as e:
                print(f"[WARN] Punctuation restoration failed: {e}")
                self.USE_PUNCTUATOR = False
                return self.basic_truecase(text)
        return text

    def basic_truecase(self, text):
        return text.capitalize()

    def apply_glossary(self, text, subs):
        for pattern, repl in subs:
            text = pattern.sub(repl, text)
        return text

    def lt_fix(self, text):
        if self.USE_LT:
            try:
                matches = self.tool.check(text)
                import language_tool_python
                return language_tool_python.utils.correct(text, matches)
            except:
                pass
        return text

    def split_for_srt(self, words, max_chars=42, max_duration=5.0, min_duration=1.0):
        lines, cur, cur_start = [], [], None
        for w in words:
            if cur_start is None:
                cur_start = w["start"]
            cur.append(w)
            text = " ".join(x["word"] for x in cur)
            duration = w["end"] - cur_start
            if len(text) >= max_chars or duration >= max_duration:
                lines.append({"start": cur_start, "end": w["end"], "text": text})
                cur, cur_start = [], None
        if cur:
            lines.append({"start": cur_start, "end": cur[-1]["end"], "text": " ".join(x["word"] for x in cur)})
        for line in lines:
            if (line["end"] - line["start"]) < min_duration:
                line["end"] = line["start"] + min_duration
        return lines

    def to_srt_time(self, t):
        h = int(t//3600); m = int((t%3600)//60); s = t%60
        return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")

    def write_srt(self, lines, path):
        with open(path, "w", encoding="utf-8") as f:
            for i, l in enumerate(lines, 1):
                f.write(f"{i}\n{self.to_srt_time(l['start'])} --> {self.to_srt_time(l['end'])}\n{l['text']}\n\n")
