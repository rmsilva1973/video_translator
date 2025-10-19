# scripts/prosody_and_ssml.py
import json, os, numpy as np
from pathlib import Path
import librosa, pyworld as pw

AUDIO = Path("work/audio/video_clean.wav")  # ou "work/video_16k_mono.wav" (escolha o que soou melhor para STT)
WORDS_JSON = Path("work/stt/video_words_aligned.json")
MT_JSON = Path("work/mt/video_en_segments.json")
OUT_JSON = Path("work/ssml/video_en_ssml.json")
OUT_SRT  = Path("work/ssml/video_en_ssml_preview.srt")
OUT_LOG  = Path("logs/ssml_report.json")

os.makedirs(OUT_JSON.parent, exist_ok=True)
os.makedirs(OUT_LOG.parent, exist_ok=True)

# 1) Carregar alinhamento e mapeamento PT->EN
aligned = json.load(open(WORDS_JSON, "r", encoding="utf-8"))
en_map  = json.load(open(MT_JSON, "r", encoding="utf-8"))
pt_segments = aligned["segments"]
en_segments = en_map["segments"]
assert len(pt_segments) == len(en_segments), "Segmentação PT e EN com comprimentos diferentes."

# 2) Carregar áudio (16k recomendado), se o arquivo não for 16k, librosa remuestre
y, sr = librosa.load(str(AUDIO), sr=16000, mono=True)
frame_len = 0.02  # 20 ms
hop_len = int(sr * frame_len)

# 3) Funções utilitárias
def segment_audio(start, end):
    s = max(0, int(start * sr))
    e = min(len(y), int(end * sr))
    return y[s:e]

def compute_rms(sig):
    if len(sig) == 0: return 0.0
    return float(np.sqrt(np.mean(sig**2) + 1e-12))

def estimate_wps(words, start, end):
    dur = max(0.01, end - start)
    wc = max(1, len(words))
    return wc / dur

def classify_rate(wps, ref_wps=(3.0, 3.5)):
    # Mantemos rate 0% no intervalo de referência; fora dele ajustamos ± até 12%
    low, high = ref_wps
    if wps < low:
        # fala mais lenta que o ideal → pode +5% ~ +10%
        delta = min(12, int((low - wps) * 6))  # heurística
        return +delta
    elif wps > high:
        # fala mais rápida que o ideal → -5% ~ -12%
        delta = min(12, int((wps - high) * 6))
        return -delta
    return 0

def compute_f0(sig):
    if len(sig) < sr*0.15:  # muito curto
        return None
    _f0, t = pw.harvest(sig.astype(np.float64), sr, f0_floor=50.0, f0_ceil=300.0)
    f0 = pw.stonemask(sig.astype(np.float64), _f0, t, sr)
    f0 = f0[f0 > 1.0]
    if len(f0) == 0:
        return None
    return f0

def classify_pitch_trend(sig):
    # Classificação muito simples: tendência da reta entre 30% e 90% do segmento
    f0 = compute_f0(sig)
    if f0 is None or len(f0) < 5:
        return "neutral"
    x = np.arange(len(f0))
    x1 = int(len(f0)*0.3)
    x2 = int(len(f0)*0.9)
    if x2 <= x1: return "neutral"
    coeff = np.polyfit(x[x1:x2], f0[x1:x2], 1)
    slope = coeff[0]
    # thresholds heurísticos
    if slope > 0.1:   # subida final
        return "question"
    if slope < -0.1:  # queda final
        return "statement"
    # ênfase (pico central) — variação relativa
    if np.max(f0) > (np.median(f0) * 1.15):
        return "emphasis"
    return "neutral"

def quantize_pause(delta_s):
    # pausas: curta/média/longa; clamp
    if delta_s < 0.15: return None
    if delta_s < 0.30: return 0.20
    if delta_s < 0.60: return 0.40
    if delta_s < 1.00: return 0.70
    return min(1.20, round(delta_s, 2))

def to_srt_time(t):
    h = int(t//3600); m = int((t%3600)//60); s = t%60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")

# 4) Varredura por segmento: pausas, rate, pitch category
out_segments = []
report = []
for i, (pt, en) in enumerate(zip(pt_segments, en_segments), start=1):
    start, end = float(pt["start"]), float(pt["end"])
    words = pt.get("words") or []
    en_text = en["en_text"]
    dur = max(0.01, end - start)

    # Pausas internas (entre palavras)
    pauses = []
    for j in range(1, len(words)):
        prev_end = float(words[j-1]["end"])
        cur_start = float(words[j]["start"])
        gap = cur_start - prev_end
        q = quantize_pause(gap)
        if q:
            # index da pausa e duração quantizada
            pauses.append({"after_index": j-1, "dur": q})

    # Velocidade (rate)
    wps = estimate_wps(words, start, end)
    rate_pct = classify_rate(wps)  # de -12 a +12 (%)

    # Pitch/entonação
    sig = segment_audio(start, end)
    pitch_cat = classify_pitch_trend(sig)
    # Mapeamento para SSML
    if pitch_cat == "question":
        pitch_ssml = "+2st"
    elif pitch_cat == "emphasis":
        pitch_ssml = "+1st"
    elif pitch_cat == "statement":
        pitch_ssml = "-1st"
    else:
        pitch_ssml = "0st"

    # Geração de SSML: pausas embutidas em texto EN por tokens aproximados (split simples)
    # Estratégia simples: dividir en_text por espaços e injetar breaks após índices proporcionais
    en_tokens = en_text.split()
    ssml_parts = []
    token_cursor = 0

    # Criar um mapeamento grosso PT->EN por proporção de índice (melhor refinaremos depois se necessário)
    def ptidx_to_enidx(pt_idx):
        if len(words) <= 1: return len(en_tokens)
        return int(round((pt_idx / (len(words)-1)) * max(0, len(en_tokens)-1)))

    for p in pauses:
        en_idx = ptidx_to_enidx(p["after_index"])
        # adicionar tokens até en_idx
        ssml_parts.append(" ".join(en_tokens[token_cursor:en_idx+1]))
        ssml_parts.append(f'<break time="{int(p["dur"]*1000)}ms"/>')
        token_cursor = en_idx + 1

    # resto dos tokens
    if token_cursor < len(en_tokens):
        ssml_parts.append(" ".join(en_tokens[token_cursor:]))

    base_text_with_breaks = " ".join([s for s in ssml_parts if s])

    # Aplicar prosody rate/pitch no nível de sentença
    # Constrói <prosody rate="+X%" pitch="+Yst"> ... </prosody>
    rate_str = f"{rate_pct}%" if rate_pct >= 0 else f"{rate_pct}%"
    ssml = f'<prosody rate="{rate_str}" pitch="{pitch_ssml}">{base_text_with_breaks}</prosody>'

    out_segments.append({
        "start": start,
        "end": end,
        "en_text": en_text,
        "ssml": ssml,
        "wps_pt": wps,
        "rate_pct": rate_pct,
        "pitch_cat": pitch_cat,
        "pauses_count": len(pauses)
    })

    report.append({
        "idx": i, "start": start, "end": end, "dur": dur,
        "wps_pt": round(wps, 2),
        "rate_pct": rate_pct,
        "pitch_cat": pitch_cat,
        "pauses": pauses[:8]  # amostra
    })

# 5) Salvar saídas
json.dump({"segments": out_segments}, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
json.dump(report, open(OUT_LOG, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

# 6) SRT “preview” com marcação textual de pausas (para revisão humana)
with open(OUT_SRT, "w", encoding="utf-8") as f:
    for idx, s in enumerate(out_segments, 1):
        text_prev = s["en_text"].replace("<", "").replace(">", "")
        # Sinaliza pausas como [pause:XXXms] apenas para revisão
        # (não é SSML real; apenas ajuda visualizar as quebras)
        preview = s["ssml"].replace("<prosody", "[prosody").replace("</prosody>", "[/prosody]").replace("<break time=\"", "[pause:").replace("\"/>", "ms]")
        f.write(f"{idx}\n{to_srt_time(s['start'])} --> {to_srt_time(s['end'])}\n{preview}\n\n")

print("OK:", OUT_JSON, OUT_SRT, "| Report:", OUT_LOG)
