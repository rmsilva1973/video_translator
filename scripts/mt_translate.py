import json, re, os, csv, hashlib
from pathlib import Path
import langid
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class MachineTranslator:
    def __init__(self, video_name):
        self.video_name = video_name
        self.pt_json = Path(f"work/stt/{Path(video_name).stem}_pt_clean.json")
        self.out_json = Path(f"work/mt/{Path(video_name).stem}_en_segments.json")
        self.out_srt = Path(f"work/mt/{Path(video_name).stem}_en.srt")
        self.out_log = Path(f"logs/{Path(video_name).stem}_mt_report.json")
        self.glossary = Path("glossary/terms.csv")
        
        os.makedirs("work/mt", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ip_re = re.compile(r"\b(\d{1,3}(?:\.\d{1,3}){3})(?:/(\d{1,2}))?\b")
        self.num_re = re.compile(r"\b\d+[\d\.,/]*\b")
        
        # Lazy loading
        self.tok_main, self.mod_main, self.en_id = None, None, None
        self.tok_fb1, self.mod_fb1 = None, None
        self.tok_fb2, self.mod_fb2 = None, None
        self.acronyms = None

    def load_models(self):
        if self.tok_main is not None:
            return
        
        self.tok_main, self.mod_main = self._try_load("facebook/nllb-200-1.3B", {"use_fast": False})
        if self.tok_main is None:
            raise RuntimeError("Failed to load NLLB-200 1.3B")
        self.en_id = self._resolve_nllb_lang_id(self.tok_main, "eng_Latn")
        
        self.tok_fb1, self.mod_fb1 = self._try_load("Helsinki-NLP/opus-mt-tc-big-pt-en")
        self.tok_fb2, self.mod_fb2 = self._try_load("facebook/m2m100_418M")
        self.acronyms = self._load_glossary_upper()

    def _try_load(self, model_name, tok_kwargs=None):
        tok_kwargs = tok_kwargs or {}
        try:
            tok = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
            mod = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else None
            )
            mod.to(self.device).eval()
            return tok, mod
        except Exception as e:
            print(f"[WARN] Failed to load {model_name}: {e}")
            return None, None

    def _resolve_nllb_lang_id(self, tok, lang_code="eng_Latn"):
        if hasattr(tok, "lang_code_to_id"):
            return tok.lang_code_to_id[lang_code]
        if hasattr(tok, "get_lang_id"):
            return tok.get_lang_id(lang_code)
        return tok.convert_tokens_to_ids(f"__{lang_code}__")

    def _load_glossary_upper(self):
        acronyms = set()
        if self.glossary.exists():
            with open(self.glossary, newline="", encoding="utf-8") as f:
                r = csv.DictReader(f, fieldnames=["find","replace","flags"])
                for row in r:
                    term = (row["find"] or "").strip()
                    if 2 <= len(term) <= 6 and term.isupper():
                        acronyms.add(term)
        base = {"S3","EC2","VPC","CIDR","IAM","TLS","TCP","UDP","VPN","DNS","NAT","SLA","VLAN","SSH","HTTP","HTTPS","ACL"}
        acronyms |= base
        return sorted(acronyms, key=len, reverse=True)

    def _protect_entities(self, text):
        placeholders = {}
        def put(val):
            key = "__TERM" + hashlib.md5(val.encode("utf-8")).hexdigest()[:8] + "__"
            placeholders[key] = val
            return key
        
        text = self.ip_re.sub(lambda m: put(m.group(0)), text)
        text = self.num_re.sub(lambda m: put(m.group(0)), text)
        for t in self.acronyms:
            patt = re.compile(rf"\b{re.escape(t)}\b")
            text = patt.sub(lambda m: put(m.group(0)), text)
        return text, placeholders

    def _restore_entities(self, text, placeholders):
        for k, v in placeholders.items():
            text = text.replace(k, v)
        return text

    def _clean_punct(self, s):
        s = re.sub(r"\s+([,\.!\?:;])", r"\1", s)
        return s.strip()

    def _is_english_like(self, text, lang, score):
        if not text or lang == "en":
            return True
        ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(1, len(text))
        return len(text) < 8 and ascii_ratio > 0.95

    def _estimate_max_tokens(self, pt_len):
        approx_tokens = max(8, int(pt_len / 4))
        return min(128, approx_tokens + 10)

    def _translate_main_nllb(self, pt_text, target_tokens):
        protected, placeholders = self._protect_entities(pt_text)
        inputs = self.tok_main(protected, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.inference_mode():
            gen = self.mod_main.generate(
                **inputs, forced_bos_token_id=self.en_id, num_beams=4,
                length_penalty=1.1, max_new_tokens=target_tokens, no_repeat_ngram_size=3
            )
        out = self.tok_main.decode(gen[0], skip_special_tokens=True)
        return self._clean_punct(self._restore_entities(out, placeholders))

    def _translate_fb_opus(self, pt_text, target_tokens):
        if self.tok_fb1 is None:
            raise RuntimeError("Opus-MT not available")
        protected, placeholders = self._protect_entities(pt_text)
        protected = ">>en<< " + protected
        inputs = self.tok_fb1(protected, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.inference_mode():
            gen = self.mod_fb1.generate(
                **inputs, num_beams=4, length_penalty=1.15,
                max_new_tokens=target_tokens, no_repeat_ngram_size=3
            )
        out = self.tok_fb1.decode(gen[0], skip_special_tokens=True)
        return self._clean_punct(self._restore_entities(out, placeholders))

    def _translate_fb_m2m(self, pt_text, target_tokens):
        if self.tok_fb2 is None:
            raise RuntimeError("M2M100 not available")
        protected, placeholders = self._protect_entities(pt_text)
        self.tok_fb2.src_lang = "pt"
        inputs = self.tok_fb2(protected, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        forced_bos = self.tok_fb2.get_lang_id("en")
        with torch.inference_mode():
            gen = self.mod_fb2.generate(
                **inputs, forced_bos_token_id=forced_bos, num_beams=4,
                length_penalty=1.15, max_new_tokens=target_tokens, no_repeat_ngram_size=3
            )
        out = self.tok_fb2.decode(gen[0], skip_special_tokens=True)
        return self._clean_punct(self._restore_entities(out, placeholders))

    def _to_srt_time(self, t):
        h = int(t//3600); m = int((t%3600)//60); s = t%60
        return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")

    def process(self):
        self.load_models()
        
        data = json.load(open(self.pt_json, "r", encoding="utf-8"))
        segments = data.get("segments", [])
        out = {"segments": []}
        report = []

        for seg in segments:
            start, end = float(seg["start"]), float(seg["end"])
            pt = (seg.get("text") or "").strip()
            dur = max(0.01, end - start)
            pt_len = len(pt)

            if not pt:
                out["segments"].append({
                    "start": start, "end": end, "pt_text": "", "en_text": "",
                    "len_ratio": 1.0, "lang": "en", "model": "none"
                })
                report.append({
                    "start": start, "end": end, "duration": dur, "pt_len": 0, "en_len": 0,
                    "len_ratio": 1.0, "lang": "en", "lang_score": 0.0,
                    "used_model": "none", "fallback_used": False, "pt": "", "en": ""
                })
                continue

            target_tokens = self._estimate_max_tokens(pt_len)
            en = self._translate_main_nllb(pt, target_tokens)
            lang, score = langid.classify(en)
            used_model = "nllb-1.3B"
            fallback_used = False

            if not self._is_english_like(en, lang, score) and self.tok_fb1:
                try:
                    en_fb = self._translate_fb_opus(pt, target_tokens)
                    lang2, score2 = langid.classify(en_fb)
                    if self._is_english_like(en_fb, lang2, score2):
                        en, lang, score = en_fb, lang2, score2
                        used_model, fallback_used = "opus-mt-pt-en", True
                except Exception as e:
                    print(f"[WARN] Opus-MT fallback failed: {e}")

            if not self._is_english_like(en, lang, score) and self.tok_fb2:
                try:
                    en_fb2 = self._translate_fb_m2m(pt, target_tokens)
                    lang3, score3 = langid.classify(en_fb2)
                    if self._is_english_like(en_fb2, lang3, score3):
                        en, lang, score = en_fb2, lang3, score3
                        used_model, fallback_used = "m2m100_418M", True
                except Exception as e:
                    print(f"[WARN] M2M100 fallback failed: {e}")

            en_len = len(en)
            len_ratio = (en_len + 1) / (pt_len + 1)

            report.append({
                "start": start, "end": end, "duration": dur,
                "pt_len": pt_len, "en_len": en_len, "len_ratio": round(len_ratio, 3),
                "lang": lang, "lang_score": round(float(score), 3),
                "used_model": used_model, "fallback_used": fallback_used,
                "pt": pt, "en": en
            })

            out["segments"].append({
                "start": start, "end": end, "pt_text": pt, "en_text": en,
                "len_ratio": float(len_ratio), "lang": lang, "model": used_model
            })

        json.dump(out, open(self.out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        json.dump(report, open(self.out_log, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

        with open(self.out_srt, "w", encoding="utf-8") as f:
            for i, s in enumerate(out["segments"], 1):
                f.write(f"{i}\n{self._to_srt_time(s['start'])} --> {self._to_srt_time(s['end'])}\n{s['en_text']}\n\n")

        print(f"OK: {self.out_json}, {self.out_srt} | Report: {self.out_log}")
