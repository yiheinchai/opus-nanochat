"""
Generates the lecture narration with Gemini TTS. Runs inside GitHub Actions (.github/workflows/tts.yml),
where the API key comes from the repository secret GEMINI_API_KEY. The key is never printed or saved.

python lecture/tts/gemini_job.py list
python lecture/tts/gemini_job.py generate --model gemini-2.5-pro-preview-tts --voice Charon --indices 0,93 --out narration
"""

import os
import sys
import json
import time
import wave
import base64
import argparse
import subprocess
import urllib.error
import urllib.request

API = "https://generativelanguage.googleapis.com/v1beta"
STYLE = "You are the narrator of a coding lecture. Read the text in a relaxed, friendly lecturer tone, at a conversational pace. Read only the text, exactly as written."

def request(method, path, body=None):
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("GEMINI_API_KEY is not set (add it as a repository secret)")
    req = urllib.request.Request(API + path, method=method, data=None if body is None else json.dumps(body).encode(),
                                 headers={"x-goog-api-key": key, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.loads(response.read())

def list_models():
    try:
        models = request("GET", "/models?pageSize=200")["models"]
    except urllib.error.HTTPError as e:
        sys.exit(f"models: HTTP {e.code} {e.read().decode(errors='replace')[:400]}")
    for m in models:
        if "tts" in m["name"]:
            print(f"  {m['name']:48s} {m.get('displayName', '')}")

def tts(text, model, voice):
    """ 24 kHz mono 16-bit PCM. The style goes in the system instruction: put in the prompt, some models read it out """
    body = {"systemInstruction": {"parts": [{"text": STYLE}]},
            "contents": [{"parts": [{"text": text}]}],
            "generationConfig": {"responseModalities": ["AUDIO"],
                                 "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}}}}
    for attempt in range(8):
        try:
            response = request("POST", f"/models/{model}:generateContent", body)
            part = response["candidates"][0]["content"]["parts"][0]
            return base64.b64decode(part["inlineData"]["data"])
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            print(f"    HTTP {e.code}: {detail}", flush=True)
            if e.code == 400 and "systemInstruction" in body: # this model takes no system instruction: no style then
                body.pop("systemInstruction")
                continue
            if e.code not in (429, 500, 502, 503, 504):
                raise
        except (KeyError, IndexError) as e: # no audio in the response, try again
            print(f"    no audio in the response ({e!r})", flush=True)
        time.sleep(min(300, 10 * 2 ** attempt))
    raise RuntimeError("giving up on this clip")

def generate(model, voice, indices, out):
    rows = [json.loads(line) for line in open(os.path.join(os.path.dirname(__file__), "manifest.jsonl"), encoding="utf-8")]
    folder = os.path.join(out, f"gemini-{model.replace('gemini-', '').replace('-preview', '')}-{voice.lower()}")
    os.makedirs(folder, exist_ok=True)
    wanted = range(len(rows)) if indices == "all" else [int(i) for i in indices.split(",")]
    print(f"voice {voice}, model {model} -> {folder}", flush=True)
    done = 0
    for i in wanted:
        row = rows[i]
        path = os.path.join(folder, row["key"] + ".mp3")
        if os.path.exists(path):
            continue
        pcm = tts(row["text"], model, voice)
        wav = path[:-4] + ".wav"
        with wave.open(wav, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(24000)
            f.writeframes(pcm)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-ar", "44100", "-ac", "1", "-b:a", "128k", path], check=True)
        os.remove(wav)
        done += 1
        print(f"  [{i}] {len(row['text'])} chars -> {os.path.basename(path)} ({len(pcm) / 48000:.1f}s)", flush=True)
    print(f"generated {done} clips in {folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["list", "generate"])
    parser.add_argument("--model", default="gemini-2.5-pro-preview-tts")
    parser.add_argument("--voice", default="Charon")
    parser.add_argument("--indices", default="0,93", help="manifest indices, or 'all'")
    parser.add_argument("--out", default="lecture/tts")
    args = parser.parse_args()
    if args.mode == "list":
        list_models()
    else:
        for voice in args.voice.split(","):
            generate(args.model, voice.strip(), args.indices, args.out)
