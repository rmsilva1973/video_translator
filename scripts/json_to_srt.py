# scripts/json_to_srt.py
import json, math

def fmt(t):
    h = int(t//3600); m = int((t%3600)//60); s = t%60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")

data = json.load(open("work/stt/video_stt.json","r",encoding="utf-8"))
with open("work/stt/video_stt.srt","w",encoding="utf-8") as out:
    for i, seg in enumerate(data["segments"], start=1):
        out.write(f"{i}\n{fmt(seg['start'])} --> {fmt(seg['end'])}\n{seg['text']}\n\n")
print("OK: work/stt/video_stt.srt gerado")
