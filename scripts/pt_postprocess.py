# scripts/pt_postprocess.py
import json, re, csv, yaml, os, sys
from pathlib import Path

# 1) PONTUAÇÃO/MAIÚSCULAS
USE_PUNCTUATOR = True
try:
    if USE_PUNCTUATOR:
        from deepmultilingualpunctuation import PunctuationModel
        punct_model = PunctuationModel()
except Exception as e:
    print(f"[WARN] Punctuation model not available: {e}")
    USE_PUNCTUATOR = False

# 2) GRAMÁTICA (LanguageTool opcional)
USE_LT = False
LT_SERVER = "http://localhost:8081"
try:
    import language_tool_python
    tool = language_tool_python.LanguageToolPublicAPI("pt-BR", LT_SERVER)
    USE_LT = True
except Exception:
    print("[INFO] LanguageTool não está ativo; seguindo sem LT local.")

def read_glossary(glossary_csv):
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

def normalize_numbers_units(text):
    # Exemplos básicos; amplie conforme seu domínio:
    # Espaços antes de unidades
    text = re.sub(r"(?i)\b(\d+)(\s*)(ghz)\b", r"\1,0 GHz", text)  # se veio sem vírgula; ajuste
    # Pontos/virgulas comuns (não agressivo)
    text = text.replace(" gigahertz", " GHz")
    text = text.replace(" megahertz", " MHz")
    text = text.replace(" gigabytes", " GB").replace(" gigabyte", " GB")
    text = text.replace(" megabytes", " MB").replace(" megabyte", " MB")
    return text

def normalize_ips(text):
    # Corrige formas faladas comuns "10 ponto 0 ponto 1" => "10.0.1"
    text = re.sub(r"(?i)\b(ponto)\b", ".", text)
    # Barra CIDR falado "barra 24" => "/24"
    text = re.sub(r"(?i)\bbarra\s*(\d{1,2})\b", r"/\1", text)
    # Remover espaços errados: "10 . 0 . 0 . 1" => "10.0.0.1"
    text = re.sub(r"\s*\.\s*", ".", text)
    return text

def apply_glossary(text, subs):
    for pattern, repl in subs:
        text = pattern.sub(repl, text)
    return text

def restore_punctuation(text):
    if not USE_PUNCTUATOR:
        return text
    # O modelo espera textos razoavelmente curtos; processe por sentenças brutas
    try:
        return punct_model.restore_punctuation(text)
    except Exception:
        return text

def basic_truecase(text):
    # Capitalizar início de sentença simples (fallback)
    text = text.strip()
    if not text: return text
    return text[0].upper() + text[1:]

def lt_fix(text):
    if not USE_LT:
        return text
    try:
        matches = tool.check(text)
        return language_tool_python.utils.correct(text, matches)
    except Exception:
        return text

def split_for_srt(words, max_chars=42, max_duration=6.0, min_duration=1.0):
    # Segmentação simples com base em tempo e comprimento; usa palavras alinhadas
    # words: [{'word': '...','start': float, 'end': float}, ...]
    lines = []
    cur = []
    cur_start = None
    for w in words:
        token = w["word"]
        if cur_start is None:
            cur_start = w["start"]
        cur.append(w)
        duration = w["end"] - cur_start
        text = " ".join(x["word"] for x in cur).strip()
        if len(text) >= max_chars or duration >= max_duration:
            lines.append({"start": cur_start, "end": w["end"], "text": text})
            cur, cur_start = [], None
    if cur:
        lines.append({"start": cur_start, "end": cur[-1]["end"], "text": " ".join(x["word"] for x in cur)})
    # Ajuste de duração mínima
    for line in lines:
        if (line["end"] - line["start"]) < min_duration:
            line["end"] = line["start"] + min_duration
    return lines

def to_srt_time(t):
    h = int(t//3600); m = int((t%3600)//60); s = t%60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")

def write_srt(lines, path):
    with open(path, "w", encoding="utf-8") as f:
        for i, l in enumerate(lines, 1):
            f.write(f"{i}\n{to_srt_time(l['start'])} --> {to_srt_time(l['end'])}\n{l['text']}\n\n")

def main():
    in_json = Path("work/stt/video_words_aligned.json")
    out_json = Path("work/stt/video_pt_clean.json")
    out_srt  = Path("work/stt/video_pt_clean.srt")
    glossary_csv = Path("glossary/terms.csv")
    itn_yaml = Path("glossary/itn_rules.yaml")  # reservado para regras futuras
    subs = read_glossary(glossary_csv)

    data = json.load(open(in_json, "r", encoding="utf-8"))
    cleaned_segments = []

    for seg in data["segments"]:
        raw = seg["text"]
        # 1) pontuação (ou heurística)
        txt = restore_punctuation(raw)
        if not USE_PUNCTUATOR:
            txt = basic_truecase(txt)

        # 2) normalizações técnicas
        txt = normalize_ips(txt)
        txt = normalize_numbers_units(txt)

        # 3) glossário
        txt = apply_glossary(txt, subs)

        # 4) correção gramatical leve (opcional)
        txt = lt_fix(txt)

        new_seg = {
            "start": seg["start"],
            "end": seg["end"],
            "text": txt.strip(),
            "words": seg.get("words", [])
        }
        cleaned_segments.append(new_seg)

    # Escrever JSON “limpo”
    os.makedirs("work/stt", exist_ok=True)
    json.dump({"segments": cleaned_segments}, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # Gerar SRT com quebras legíveis usando palavras alinhadas (se existirem)
    lines = []
    for seg in cleaned_segments:
        words = seg.get("words") or []
        if words:
            # Refaça as palavras a partir do texto limpo se necessário (simples):
            # fallback: distribui o texto no intervalo original quando words estiver vazio
            lines.extend(split_for_srt(words))
        else:
            lines.append({"start": seg["start"], "end": seg["end"], "text": seg["text"]})

    write_srt(lines, out_srt)
    print("OK:", out_json, out_srt)

if __name__ == "__main__":
    main()
