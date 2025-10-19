# scripts/mt_translate.py
import json, re, os, csv, hashlib
from pathlib import Path
import langid
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

PT_JSON = Path("work/stt/video_pt_clean.json")
OUT_JSON = Path("work/mt/video_en_segments.json")
OUT_SRT  = Path("work/mt/video_en.srt")
OUT_LOG  = Path("logs/mt_report.json")
GLOSSARY = Path("glossary/terms.csv")

# Pastas
os.makedirs("work/mt", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# =========================
# Helpers de modelo/device
# =========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def try_load(model_name, tok_kwargs=None):
    tok_kwargs = tok_kwargs or {}
    try:
        tok = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
        mod = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if DEVICE == "cuda" else None
        )
        mod.to(DEVICE)
        mod.eval()
        return tok, mod
    except Exception as e:
        print(f"[WARN] Falha ao carregar {model_name}: {e}")
        return None, None

def resolve_nllb_lang_id(tok, lang_code="eng_Latn"):
    # 1) NLLB slow
    if hasattr(tok, "lang_code_to_id"):
        return tok.lang_code_to_id[lang_code]
    # 2) NLLB fast novas versões
    if hasattr(tok, "get_lang_id"):
        return tok.get_lang_id(lang_code)
    # 3) Token explícito
    return tok.convert_tokens_to_ids(f"__{lang_code}__")


# =========================
# Modelos (principal + fallback)
# =========================
# Principal: NLLB-200 1.3B (melhor qualidade, força inglês com forced_bos_token_id)
MODEL_MAIN = "facebook/nllb-200-1.3B"
tok_main, mod_main = try_load(MODEL_MAIN, tok_kwargs={"use_fast": False})
if tok_main is None or mod_main is None:
    raise RuntimeError("Não foi possível carregar o modelo principal NLLB-200 1.3B.")
EN_ID = resolve_nllb_lang_id(tok_main, "eng_Latn")

# Fallback 1: Marian PT->EN (estável)
MODEL_FB1 = "Helsinki-NLP/opus-mt-tc-big-pt-en"
tok_fb1, mod_fb1 = try_load(MODEL_FB1)

# Fallback 2 (opcional): M2M100 418M
MODEL_FB2 = "facebook/m2m100_418M"
tok_fb2, mod_fb2 = try_load(MODEL_FB2)

# =========================
# Placeholders conservadores (IPs, números e SIGLAS UPPERCASE)
# =========================
IP_RE   = re.compile(r"\b(\d{1,3}(?:\.\d{1,3}){3})(?:/(\d{1,2}))?\b")
NUM_RE  = re.compile(r"\b\d+[\d\.,/]*\b")

def load_glossary_upper(csv_path):
    """
    Carrega termos do glossário, mantendo apenas SIGLAS UPPERCASE curtas (2–6 chars),
    para evitar excesso de placeholders em nomes próprios (ex.: Kubernetes).
    """
    acronyms = set()
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f, fieldnames=["find","replace","flags"])
            for row in r:
                term = (row["find"] or "").strip()
                if 2 <= len(term) <= 6 and term.isupper():
                    acronyms.add(term)
    # Base de acrônimos comuns em TI (maiúsculos)
    base = {"S3","EC2","VPC","CIDR","IAM","TLS","TCP","UDP","VPN","DNS","NAT","SLA","VLAN","SSH","HTTP","HTTPS","ACL"}
    acronyms |= base
    # Ordena por tamanho decrescente (evita sobreposição)
    return sorted(acronyms, key=len, reverse=True)

ACRONYMS = load_glossary_upper(GLOSSARY)

def protect_entities(text):
    """
    Protege IP/CIDR, números e SIGLAS UPPERCASE com placeholders.
    Mantém o restante do texto intacto para preservar contexto do NMT.
    """
    placeholders = {}

    def put(val):
        key = "__TERM" + hashlib.md5(val.encode("utf-8")).hexdigest()[:8] + "__"
        placeholders[key] = val
        return key

    # IP/CIDR
    text = IP_RE.sub(lambda m: put(m.group(0)), text)
    # Números
    text = NUM_RE.sub(lambda m: put(m.group(0)), text)
    # Acrônimos (case-sensitive)
    for t in ACRONYMS:
        patt = re.compile(rf"\b{re.escape(t)}\b")
        text = patt.sub(lambda m: put(m.group(0)), text)

    return text, placeholders

def restore_entities(text, placeholders):
    for k, v in placeholders.items():
        text = text.replace(k, v)
    return text

# =========================
# Utilitários
# =========================
def to_srt_time(t):
    h = int(t//3600); m = int((t%3600)//60); s = t%60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")

def estimate_max_tokens(pt_len):
    """
    Estima max_new_tokens a partir do comprimento do PT.
    Heurística conservadora para evitar verbosidade excessiva.
    """
    approx_tokens = max(8, int(pt_len / 4))  # ~4 chars por token (grosseiro)
    return min(128, approx_tokens + 10)

def clean_punct(s):
    s = re.sub(r"\s+([,\.!\?:;])", r"\1", s)
    return s.strip()

def is_english_like(text, lang, score):
    """
    Heurística leve para validar saída em EN:
    - aceita se langid disser 'en'
    - para textos muito curtos, aceita se a razão ASCII for alta
    """
    if not text:
        return True
    if lang == "en":
        return True
    ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(1, len(text))
    return len(text) < 8 and ascii_ratio > 0.95

# =========================
# Tradução: principal (NLLB) + fallbacks
# =========================
def translate_main_nllb(pt_text, target_tokens):
    protected, placeholders = protect_entities(pt_text)
    inputs = tok_main(protected, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.inference_mode():
        gen = mod_main.generate(
            **inputs,
            forced_bos_token_id=EN_ID,
            num_beams=4,
            length_penalty=1.1,
            max_new_tokens=target_tokens,
            no_repeat_ngram_size=3
        )
    out = tok_main.decode(gen[0], skip_special_tokens=True)
    out = restore_entities(out, placeholders)
    return clean_punct(out)

def translate_fb_m2m(pt_text, target_tokens):
    if tok_fb2 is None or mod_fb2 is None:
        raise RuntimeError("Fallback M2M100 não está disponível.")
    protected, placeholders = protect_entities(pt_text)
    tok_fb2.src_lang = "pt"  # origem PT
    inputs = tok_fb2(protected, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    forced_bos = tok_fb2.get_lang_id("en")
    with torch.inference_mode():
        gen = mod_fb2.generate(
            **inputs,
            forced_bos_token_id=forced_bos,
            num_beams=4,
            length_penalty=1.15,
            max_new_tokens=target_tokens,
            no_repeat_ngram_size=3
        )
    out = tok_fb2.decode(gen[0], skip_special_tokens=True)
    out = restore_entities(out, placeholders)
    return clean_punct(out)

def translate_fb_opus(pt_text, target_tokens):
    if tok_fb1 is None or mod_fb1 is None:
        raise RuntimeError("Fallback Marian não está disponível.")
    protected, placeholders = protect_entities(pt_text)
    protected = ">>en<< " + protected
    inputs = tok_fb1(protected, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.inference_mode():
        gen = mod_fb1.generate(
            **inputs,
            num_beams=4,
            length_penalty=1.15,
            max_new_tokens=target_tokens,
            no_repeat_ngram_size=3
        )
    out = tok_fb1.decode(gen[0], skip_special_tokens=True)
    out = restore_entities(out, placeholders)
    return clean_punct(out)

# =========================
# Processo principal
# =========================
data = json.load(open(PT_JSON, "r", encoding="utf-8"))
segments = data.get("segments", [])
out = {"segments": []}
report = []

for seg in segments:
    start, end = float(seg["start"]), float(seg["end"])
    pt = (seg.get("text") or "").strip()
    dur = max(0.01, end - start)
    pt_len = len(pt)

    # Segmentos vazios: pule geração e registre saneado
    if not pt:
        out["segments"].append({
            "start": start, "end": end,
            "pt_text": "", "en_text": "",
            "len_ratio": 1.0,
            "lang": "en",
            "model": "none"
        })
        report.append({
            "start": start, "end": end, "duration": dur,
            "pt_len": 0, "en_len": 0, "len_ratio": 1.0,
            "lang": "en", "lang_score": 0.0,
            "used_model": "none", "fallback_used": False,
            "pt": "", "en": ""
        })
        continue

    # Estima tokens-alvo
    target_tokens = estimate_max_tokens(pt_len)

    # 1) Tenta NLLB (principal)
    en = translate_main_nllb(pt, target_tokens)
    lang, score = langid.classify(en)

    used_model = "nllb-1.3B"
    fallback_used = False

    ok_en = is_english_like(en, lang, score)

    # 2) Se não saiu EN, tenta fallback OPUS-MT
    if not ok_en and tok_fb1 and mod_fb1:
        try:
            en_fb = translate_fb_opus(pt, target_tokens)
            lang2, score2 = langid.classify(en_fb)
            if is_english_like(en_fb, lang2, score2):
                en = en_fb
                lang, score = lang2, score2
                used_model = "opus-mt-pt-en"
                fallback_used = True
        except Exception as e:
            print(f"[WARN] Fallback Marian falhou: {e}")

    # 3) Se ainda não saiu EN, tenta M2M100
    if not ok_en and tok_fb2 and mod_fb2:
        try:
            en_fb2 = translate_fb_m2m(pt, target_tokens)
            lang3, score3 = langid.classify(en_fb2)
            if is_english_like(en_fb2, lang3, score3):
                en = en_fb2
                lang, score = lang3, score3
                used_model = "m2m100_418M"
                fallback_used = True
        except Exception as e:
            print(f"[WARN] Fallback M2M100 falhou: {e}")

    # Recalcula ok_en após possíveis fallbacks (para consistência)
    ok_en = is_english_like(en, lang, score)
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
        "start": start, "end": end,
        "pt_text": pt,
        "en_text": en,
        "len_ratio": float(len_ratio),
        "lang": lang,
        "model": used_model
    })

# Escrita de saídas
json.dump(out, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
json.dump(report, open(OUT_LOG, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

# SRT EN provisório (usa tempos do PT e o texto en_text)
with open(OUT_SRT, "w", encoding="utf-8") as f:
    for i, s in enumerate(out["segments"], 1):
        f.write(f"{i}\n{to_srt_time(s['start'])} --> {to_srt_time(s['end'])}\n{s['en_text']}\n\n")

print("OK:", OUT_JSON, OUT_SRT, "| Relatório:", OUT_LOG)