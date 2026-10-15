"""
Generates the lecture narration with ElevenLabs. Runs inside GitHub Actions (.github/workflows/tts.yml),
where the API key comes from the repository secret ELEVENLABS_API_KEY. The key is never printed or saved.

python lecture/tts/elevenlabs_job.py list
python lecture/tts/elevenlabs_job.py generate --model eleven_v3 --voice <voice_id> --indices 0,70,75,93 --out lecture/tts
"""

import os
import re
import sys
import json
import time
import argparse
import urllib.error
import urllib.request

API = "https://api.elevenlabs.io"

def request(method, path, body=None, accept="application/json"):
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        sys.exit("ELEVENLABS_API_KEY is not set (add it as a repository secret)")
    req = urllib.request.Request(API + path, method=method, data=None if body is None else json.dumps(body).encode(),
                                 headers={"xi-api-key": key, "Content-Type": "application/json", "Accept": accept})
    with urllib.request.urlopen(req, timeout=300) as response:
        return response.read()

def list_everything():
    sub = json.loads(request("GET", "/v1/user/subscription"))
    print(f"subscription: tier={sub.get('tier')} characters used {sub.get('character_count')} of {sub.get('character_limit')}")
    print("\nMODELS")
    for m in json.loads(request("GET", "/v1/models")):
        if m.get("can_do_text_to_speech"):
            print(f"  {m['model_id']:32s} {m.get('name')}  (max chars/request: {m.get('maximum_text_length_per_request')})")
    print("\nVOICES")
    for v in json.loads(request("GET", "/v1/voices"))["voices"]:
        labels = ", ".join(f"{k}={val}" for k, val in (v.get("labels") or {}).items())
        print(f"  {v['voice_id']}  {v['name']:24s} {v.get('category', ''):10s} {labels}")

def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")

def tts(text, model, voice, previous_text, next_text):
    body = {"text": text, "model_id": model, "previous_text": previous_text, "next_text": next_text,
            "voice_settings": {"stability": 0.5} if model.startswith("eleven_v3") else
                              {"stability": 0.45, "similarity_boost": 0.75, "style": 0.2, "use_speaker_boost": True}}
    attempts = [body, {k: v for k, v in body.items() if k not in ("previous_text", "next_text")},
                {k: v for k, v in body.items() if k not in ("previous_text", "next_text", "voice_settings")}]
    for attempt in range(6):
        payload = attempts[min(attempt, len(attempts) - 1)] if attempt < 3 else attempts[-1]
        try:
            return request("POST", f"/v1/text-to-speech/{voice}?output_format=mp3_44100_128", payload, accept="audio/mpeg")
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            print(f"    HTTP {e.code}: {detail}", flush=True)
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt * 5)
            elif e.code != 400 and e.code != 422:
                raise
    raise RuntimeError("giving up on this clip")

def generate(model, voice, indices, out):
    rows = [json.loads(line) for line in open(os.path.join(os.path.dirname(__file__), "manifest.jsonl"), encoding="utf-8")]
    name = json.loads(request("GET", f"/v1/voices/{voice}")).get("name", voice)
    folder = os.path.join(out, f"elevenlabs-{slug(model)}-{slug(name)}")
    os.makedirs(folder, exist_ok=True)
    wanted = range(len(rows)) if indices == "all" else [int(i) for i in indices.split(",")]
    print(f"voice {name} ({voice}), model {model} -> {folder}", flush=True)
    done = 0
    for i in wanted:
        row = rows[i]
        path = os.path.join(folder, row["key"] + ".mp3")
        if os.path.exists(path):
            continue # already made, maybe in an earlier run
        audio = tts(row["text"], model, voice, rows[i - 1]["text"] if i > 0 else None, rows[i + 1]["text"] if i + 1 < len(rows) else None)
        with open(path, "wb") as f:
            f.write(audio)
        done += 1
        print(f"  [{i}] {len(row['text'])} chars -> {os.path.basename(path)} ({len(audio) // 1024} KB)", flush=True)
    print(f"generated {done} clips in {folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["list", "generate"])
    parser.add_argument("--model", default="eleven_v3")
    parser.add_argument("--voice", help="voice id (comma separated for several)")
    parser.add_argument("--indices", default="0,70,75,93", help="manifest indices, or 'all'")
    parser.add_argument("--out", default="lecture/tts")
    args = parser.parse_args()
    if args.mode == "list":
        list_everything()
    else:
        for voice in args.voice.split(","):
            generate(args.model, voice.strip(), args.indices, args.out)
