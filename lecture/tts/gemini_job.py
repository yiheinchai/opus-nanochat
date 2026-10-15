"""
Generates the lecture narration with Gemini TTS. Runs inside GitHub Actions (.github/workflows/tts.yml),
where the API key comes from the repository secret GEMINI_API_KEY. The key is never printed or saved.

python lecture/tts/gemini_job.py list
python lecture/tts/gemini_job.py generate --model gemini-3.8-flash-tts --voice Charon --indices 0,93 --out narration
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
    """ 24 kHz mono 16-bit PCM. No style prompt: in the text some models read it out, and the TTS models take no system instruction """
    body = {"contents": [{"parts": [{"text": text}]}],
            "generationConfig": {"responseModalities": ["AUDIO"],
                                 "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}}}}
    for attempt in range(6):
        try:
            response = request("POST", f"/models/{model}:generateContent", body)
            part = response["candidates"][0]["content"]["parts"][0]
            return base64.b64decode(part["inlineData"]["data"])
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            print(f"    HTTP {e.code}: {detail[:1500]}", flush=True)
            if e.code == 429 and "PerDay" in detail:
                sys.exit("daily quota used up: stopping (what's done so far still gets uploaded)")
            if e.code not in (429, 500, 502, 503, 504):
                raise
        except (KeyError, IndexError) as e: # no audio in the response, try again
            print(f"    no audio in the response ({e!r})", flush=True)
        time.sleep(min(120, 15 * 2 ** attempt))
    raise RuntimeError("giving up")

def save_mp3(pcm, path):
    wav = path[:-4] + ".wav"
    with wave.open(wav, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(24000)
        f.writeframes(pcm)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-ar", "44100", "-ac", "1", "-b:a", "128k", path], check=True)
    os.remove(wav)

def folder_for(out, model, voice):
    return os.path.join(out, f"gemini-{model.replace('gemini-', '').replace('-preview', '')}-{voice.lower()}")

def generate(model, voice, indices, out):
    """ one request per clip of the manifest """
    rows = [json.loads(line) for line in open(os.path.join(os.path.dirname(__file__), "manifest.jsonl"), encoding="utf-8")]
    folder = folder_for(out, model, voice)
    os.makedirs(folder, exist_ok=True)
    wanted = range(len(rows)) if indices == "all" else [int(i) for i in indices.split(",")]
    print(f"voice {voice}, model {model} -> {folder}", flush=True)
    for i in wanted:
        row = rows[i]
        path = os.path.join(folder, row["key"] + ".mp3")
        if not os.path.exists(path):
            pcm = tts(row["text"], model, voice)
            save_mp3(pcm, path)
            print(f"  [{i}] {len(row['text'])} chars -> {os.path.basename(path)} ({len(pcm) / 48000:.1f}s)", flush=True)

def generate_parts(model, voice, parts, out):
    """ one request per part of lecture/tts/script (a few minutes each): far fewer requests. Split them with build.py import """
    folder = folder_for(out, model, voice) + "-parts"
    os.makedirs(folder, exist_ok=True)
    script_dir = os.path.join(os.path.dirname(__file__), "script")
    names = sorted(n[:-4] for n in os.listdir(script_dir)) if parts == "all" else parts.split(",")
    print(f"voice {voice}, model {model}, {len(names)} parts -> {folder}", flush=True)
    for name in names:
        path = os.path.join(folder, name + ".mp3")
        text = open(os.path.join(script_dir, name + ".txt"), encoding="utf-8").read().strip()
        pcm = tts(text, model, voice)
        save_mp3(pcm, path)
        print(f"  {name}: {len(text)} chars -> {len(pcm) / 48000:.1f}s", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["list", "generate"])
    parser.add_argument("--model", default="gemini-3.8-flash-tts")
    parser.add_argument("--voice", default="Charon")
    parser.add_argument("--indices", default="0,93", help="manifest indices, or 'all'")
    parser.add_argument("--parts", default=None, help="instead of clips: script parts (comma separated names, or 'all')")
    parser.add_argument("--out", default="lecture/tts")
    args = parser.parse_args()
    if args.mode == "list":
        list_models()
    else:
        for voice in args.voice.split(","):
            if args.parts:
                generate_parts(args.model, voice.strip(), args.parts, args.out)
            else:
                generate(args.model, voice.strip(), args.indices, args.out)
