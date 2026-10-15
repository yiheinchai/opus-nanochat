"""
Turns lecture/lecture.md into the follow-along notebook, the narration, and the video.

python lecture/build.py notebook --execute   # posttraining.ipynb, with real outputs (runs the python cells)
python lecture/build.py audio                # narration with Kokoro-82M, locally (pip install kokoro soundfile)
python lecture/build.py audio --tts=elevenlabs   # or ElevenLabs (needs ELEVENLABS_API_KEY), or --tts=fish for Drama 3
python lecture/build.py video                # lecture/build/lecture.mp4 + lecture.srt (same --tts as the audio step)
python lecture/build.py all

python lecture/build.py audio --tts=silent   # silent placeholder audio, to preview the video timing
python lecture/build.py video --chapters=2   # a quick preview of just the first two chapters
"""

import os
import re
import sys
import json
import shlex
import hashlib
import argparse
import textwrap
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BUILD = os.path.join(HERE, "build")
NOTEBOOK = os.path.join(ROOT, "posttraining.ipynb")

# -----------------------------------------------------------------------------
# Parsing the lecture

def extract(filename, start, end=None):
    """ The lines of a repo file from the first line containing `start`, up to (not including) the next line containing `end` """
    with open(os.path.join(ROOT, filename), encoding="utf-8") as f:
        lines = f.read().split("\n")
    i0 = next(i for i, line in enumerate(lines) if start in line)
    i1 = next((i for i in range(i0 + 1, len(lines)) if end is not None and end in lines[i]), len(lines))
    return textwrap.dedent("\n".join(lines[i0:i1])).strip("\n")

def parse(path=os.path.join(HERE, "lecture.md")):
    """ Returns a list of blocks: title | chapter | narration | exercise | code | bash | output """
    with open(path, encoding="utf-8") as f:
        text = re.sub(r"<!--.*?-->", "", f.read(), flags=re.S)
    blocks, lines, i = [], text.split("\n"), 0
    paragraph = []
    def flush():
        if paragraph:
            blocks.append({"type": "narration", "text": " ".join(paragraph).strip()})
            paragraph.clear()
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            flush()
            info = shlex.split(line[3:])
            kind, opts = info[0], dict(kv.split("=", 1) for kv in info[1:])
            body = []
            i += 1
            while not lines[i].startswith("```"):
                body.append(lines[i])
                i += 1
            code = "\n".join(body).strip("\n")
            if kind == "python":
                if "from" in opts:
                    code = extract(opts["from"], opts["start"], opts.get("end"))
                blocks.append({"type": "code", "code": code, "source": opts.get("from"), "exec": opts.get("exec", "1") == "1"})
            elif kind == "bash":
                blocks.append({"type": "bash", "code": code, "exec": opts.get("exec", "1") == "1"})
            elif kind == "output":
                blocks.append({"type": "output", "text": code})
        elif line.startswith("# "):
            flush()
            blocks.append({"type": "title", "text": line[2:].strip()})
        elif line.startswith("## "):
            flush()
            blocks.append({"type": "chapter", "text": line[3:].strip()})
        elif line.startswith("> **Exercise:**"):
            flush()
            blocks.append({"type": "exercise", "text": line[len("> **Exercise:**"):].strip()})
        elif line.strip() == "":
            flush()
        else:
            paragraph.append(line.strip())
        i += 1
    flush()
    return blocks

# -----------------------------------------------------------------------------
# The notebook

COLAB_SETUP = """\
# Setup. On Colab (Runtime -> Change runtime type -> GPU), this clones the repo and installs the dependencies.
import os, sys
if "google.colab" in sys.modules and not os.path.exists("gpt.py"):
    !git clone https://github.com/yiheinchai/opus-nanochat.git
    %cd opus-nanochat
    !pip install -q tiktoken safetensors pyarrow"""

def build_notebook(blocks, execute=False):
    import nbformat
    from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
    cells, executable = [new_code_cell(COLAB_SETUP)], []
    for b in blocks:
        if b["type"] == "title":
            cells.append(new_markdown_cell(f"# {b['text']}\n\nA lecture in the style of Andrej Karpathy's *Neural Networks: Zero to Hero*, "
                                           f"following the post-training pipeline of [nanochat](https://github.com/karpathy/nanochat)."))
        elif b["type"] == "chapter":
            cells.append(new_markdown_cell(f"## {b['text']}"))
        elif b["type"] == "narration":
            cells.append(new_markdown_cell(b["text"]))
        elif b["type"] == "exercise":
            cells.append(new_markdown_cell(f"> **Exercise:** {b['text']}"))
        elif b["type"] == "code":
            header = f"# from {b['source']}\n" if b["source"] else ""
            if b["exec"]:
                cells.append(new_code_cell(header + b["code"]))
                executable.append(len(cells) - 1)
            else:
                cells.append(new_markdown_cell(f"`{b['source']}`\n```python\n{b['code']}\n```"))
        elif b["type"] == "bash" and b["exec"]:
            cells.append(new_code_cell("\n".join("!" + line for line in b["code"].split("\n") if not line.startswith("#"))))
        elif b["type"] == "bash":
            cells.append(new_markdown_cell(f"```bash\n{b['code']}\n```"))
        elif b["type"] == "output":
            cells.append(new_markdown_cell(f"Output of the lecture's run:\n```\n{b['text']}\n```"))
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    if execute:
        # execute only the python cells (the bash cells are long training runs, their outputs come from the lecture)
        from nbclient import NotebookClient
        exec_nb = new_notebook(cells=[nb.cells[i] for i in executable], metadata=nb.metadata)
        NotebookClient(exec_nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": ROOT}}).execute()
        for i, cell in zip(executable, exec_nb.cells):
            nb.cells[i] = cell
    nbformat.write(nb, NOTEBOOK)
    print(f"wrote {NOTEBOOK}")

def notebook_outputs():
    """ code -> text output, from the executed notebook (if there is one) """
    if not os.path.exists(NOTEBOOK):
        return {}
    with open(NOTEBOOK, encoding="utf-8") as f:
        nb = json.load(f)
    outputs = {}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        text = ""
        for o in cell.get("outputs", []):
            if o["output_type"] == "stream":
                text += "".join(o["text"])
            elif o["output_type"] in ("execute_result", "display_data"):
                text += "".join(o["data"].get("text/plain", ""))
            elif o["output_type"] == "error":
                text += "\n".join(o["traceback"])
        source = "".join(cell["source"])
        source = re.sub(r"^# from \S+\n", "", source)
        outputs[source] = text.rstrip("\n")
    return outputs

# -----------------------------------------------------------------------------
# The narration

# how to say things out loud
PRONUNCIATIONS = [
    ("GSM8K", "G S M 8 K"), ("MMLU", "M M L U"), ("ChatCORE", "Chat Core"), ("nanochat", "nano chat"),
    ("nanoGPT", "nano G P T"), ("GPT-2", "G P T 2"), ("SFT", "S F T"), ("GRPO", "G R P O"), ("RLHF", "R L H F"),
    ("DPO", "D P O"), ("PPO", "P P O"), ("DAPO", "Dapo"), ("KV", "K V"), ("AdamW", "Adam W"), ("tiktoken", "tick token"),
    ("ChatML", "Chat M L"), ("LLMs", "L L Ms"), ("LLM", "L L M"), ("HuggingFace", "Hugging Face"), ("makemore", "make more"),
    ("micrograd", "micro grad"), ("SmolTalk", "Smol Talk"), ("A100", "A 100"), ("H100s", "H 100s"), ("H100", "H 100"),
    ("bfloat16", "B float 16"), ("chat_eval", "chat eval"), ("chat_sft", "chat S F T"), ("chat_rl", "chat R L"),
    ("chat_cli", "chat C L I"), ("render_conversation", "render conversation"), ("loss_reduction", "loss reduction"),
    ("gpt.py", "the G P T dot pie file"), ("tasks.py", "tasks dot pie"), ("engine.py", "engine dot pie"),
    ("tokenizer.py", "tokenizer dot pie"), ("pass@k", "pass at k"), ("pass@1", "pass at 1"), ("nan", "NaN"),
]

def spoken(text):
    text = re.sub(r"[`*]", "", text) # markdown emphasis and code
    for written, said in PRONUNCIATIONS:
        text = re.sub(rf"(?<![\w-]){re.escape(written)}(?![\w-])", said, text)
    return text.replace("_", " ")

def narration_segments(blocks):
    """ Every piece of spoken text, in order """
    segments = []
    for b in blocks:
        if b["type"] == "narration":
            segments.append(spoken(b["text"]))
        elif b["type"] == "exercise":
            segments.append(spoken("Okay, here's an exercise for you. " + b["text"] + " Pause the video here and give it a try."))
    return segments

# text-to-speech backends: name -> (default voice, model)
BACKENDS = {
    "kokoro": ("am_michael", "kokoro-82m"),                         # open weights, runs on a CPU, no account needed
    "elevenlabs": ("JBFqnCBsd6RMkjVDRZzb", "eleven_multilingual_v2"), # hosted, needs ELEVENLABS_API_KEY
    "fish": ("default", "drama-3"),                                  # Fish Audio Drama 3 on your own (big) GPU server
    "silent": ("silent", "silent"),                                  # placeholder audio, to preview the video
}

# Kokoro lets us spell out the pronunciation of names it gets wrong (misaki phonemes)
KOKORO_PRONUNCIATIONS = [("Andrej", "[Andrej](/ˈɑndɹA/)"), ("Karpathy's", "[Karpathy's](/kɑɹpˈɑθiz/)"), ("Karpathy", "[Karpathy](/kɑɹpˈɑθi/)")]

def audio_path(text, tts, voice):
    key = hashlib.sha1(f"{tts}|{voice}|{BACKENDS[tts][1]}|{text}".encode()).hexdigest()[:16]
    return os.path.join(BUILD, "audio", f"{key}.mp3")

def to_mp3(src, path):
    # every clip ends up as the same kind of mp3, so that they can be concatenated
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-ar", "44100", "-ac", "1", "-b:a", "128k", path], check=True)

def tts_kokoro(text, path, voice):
    import numpy as np
    import soundfile as sf
    global _kokoro
    if "_kokoro" not in globals():
        from kokoro import KPipeline
        _kokoro = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M") # American English
    for written, said in KOKORO_PRONUNCIATIONS:
        text = re.sub(rf"\b{re.escape(written)}(?![\w'])", said, text)
    audio = np.concatenate([chunk for _, _, chunk in _kokoro(text, voice=voice)])
    sf.write(path + ".wav", audio, 24000)
    to_mp3(path + ".wav", path)
    os.remove(path + ".wav")

def tts_fish(text, path, voice):
    """
    Fish Audio Drama 3, served with sglang-omni on your own GPU (it needs an A100/H100 80GB, see the README).
    NOTE: untested, written against the /v1/audio/speech endpoint described on the model card.
    """
    import requests
    url = os.environ.get("FISH_TTS_URL", "http://localhost:8000/v1/audio/speech")
    response = requests.post(url, json={"input": text}, timeout=600)
    if response.status_code != 200:
        sys.exit(f"Drama 3 server error {response.status_code}: {response.text[:500]}")
    with open(path + ".raw", "wb") as f:
        f.write(response.content)
    to_mp3(path + ".raw", path)
    os.remove(path + ".raw")

def tts_elevenlabs(text, path, voice, previous_text, next_text):
    import requests
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        sys.exit("Set the ELEVENLABS_API_KEY environment variable (see the README), or use --tts=kokoro")
    response = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice}",
        params={"output_format": "mp3_44100_128"},
        headers={"xi-api-key": api_key, "Content-Type": "application/json"},
        json={
            "text": text,
            "model_id": BACKENDS["elevenlabs"][1],
            "previous_text": previous_text, # gives the voice continuity across paragraphs
            "next_text": next_text,
            "voice_settings": {"stability": 0.45, "similarity_boost": 0.75, "style": 0.15, "use_speaker_boost": True},
        },
        timeout=300,
    )
    if response.status_code != 200:
        sys.exit(f"ElevenLabs error {response.status_code}: {response.text[:500]}")
    with open(path, "wb") as f:
        f.write(response.content)

def tts_silent(text, path):
    seconds = len(text) / 15 # roughly how fast people talk, ~15 characters per second
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
                    "-t", f"{seconds:.2f}", "-b:a", "64k", path], check=True)

def build_audio(blocks, tts, voice):
    segments = narration_segments(blocks)
    os.makedirs(os.path.join(BUILD, "audio"), exist_ok=True)
    print(f"{len(segments)} narration segments, {sum(len(s) for s in segments):,} characters")
    for i, text in enumerate(segments):
        path = audio_path(text, tts, voice)
        if os.path.exists(path):
            continue # already generated: every clip is cached, so edits only re-synthesize what changed
        if tts == "kokoro":
            tts_kokoro(text, path, voice)
        elif tts == "fish":
            tts_fish(text, path, voice)
        elif tts == "elevenlabs":
            tts_elevenlabs(text, path, voice, segments[i - 1] if i > 0 else None, segments[i + 1] if i + 1 < len(segments) else None)
        else:
            tts_silent(text, path)
        print(f"\r[{i + 1}/{len(segments)}] {path}", end="", flush=True)
    print()

def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True, check=True)
    return float(out.stdout.strip())

# -----------------------------------------------------------------------------
# The video: a "screen" (code, its output, or a title card) shown while the narration plays

W, H = 1920, 1080
FONT_DIR = "/usr/share/fonts/truetype/dejavu"
COLORS = {"bg": "#ffffff", "header": "#f3f3f3", "text": "#1f1f1f", "muted": "#7a7a7a",
          "codebg": "#fafafa", "outbg": "#f1f3f5", "border": "#dddddd", "accent": "#c25e00"}
ANSI = {"91": "#d32f2f", "92": "#2e7d32", "90": "#9e9e9e", "0": None}

def font(size, bold=False, mono=False):
    from PIL import ImageFont
    name = "DejaVuSansMono" if mono else "DejaVuSans"
    candidates = [os.path.join(FONT_DIR, f"{name}{'-Bold' if bold else ''}.ttf"), f"{name}{'-Bold' if bold else ''}.ttf"]
    for c in candidates:
        try:
            return ImageFont.truetype(c, size)
        except OSError:
            pass
    return ImageFont.load_default(size)

def highlight(code, language):
    """ code -> list of lines, each a list of (text, color) """
    from pygments import lex
    from pygments.lexers import PythonLexer, BashLexer
    from pygments.styles import get_style_by_name
    style = get_style_by_name("friendly")
    lines = [[]]
    for ttype, value in lex(code, PythonLexer() if language == "python" else BashLexer()):
        color = style.style_for_token(ttype)["color"]
        color = f"#{color}" if color else COLORS["text"]
        for j, piece in enumerate(value.split("\n")):
            if j > 0:
                lines.append([])
            if piece:
                lines[-1].append((piece, color))
    while lines and not lines[-1]:
        lines.pop()
    return lines

def ansi_lines(text):
    """ output text (possibly with ANSI colors) -> list of lines, each a list of (text, color) """
    lines, color = [[]], None
    for piece in re.split(r"(\x1b\[\d+m)", text):
        m = re.fullmatch(r"\x1b\[(\d+)m", piece)
        if m:
            color = ANSI.get(m.group(1))
            continue
        for j, chunk in enumerate(piece.split("\n")):
            if j > 0:
                lines.append([])
            if chunk:
                lines[-1].append((chunk, color or COLORS["text"]))
    return lines

def wrap(lines, max_chars):
    out = []
    for line in lines:
        current, n = [], 0
        for text, color in line:
            while n + len(text) > max_chars:
                take = max_chars - n
                current.append((text[:take], color))
                out.append(current)
                current, n, text = [], 0, text[take:]
            current.append((text, color))
            n += len(text)
        out.append(current)
    return out

def draw_lines(draw, lines, x, y, fnt, line_height):
    for line in lines:
        cx = x
        for text, color in line:
            draw.text((cx, y), text, font=fnt, fill=color)
            cx += draw.textlength(text, font=fnt)
        y += line_height
    return y

def layout(screen):
    """
    Lay out code + output in pages. Use the biggest font (down to MIN_FONT) at which everything
    fits on one screen, otherwise split the code over several pages (shown one after another).
    Returns the font size and a list of pages, each page is (code lines, output lines)
    """
    from PIL import Image, ImageDraw
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    code = highlight(screen["code"], screen["language"])
    output = ansi_lines(screen["output"]) if screen.get("output") else []
    height = (H - 30) - 100
    for size in range(26, MIN_FONT - 1, -1):
        max_chars = int((W - 140) / draw.textlength("x", font=font(size, mono=True)))
        code_w, out_w = wrap(code, max_chars), wrap(output, max_chars)
        line_h = int(size * 1.4)
        if len(code_w) * line_h + 40 + (len(out_w) * line_h + 70 if out_w else 0) <= height:
            return size, [(code_w, out_w)]
    lines_per_page = (height - 40) // line_h
    pages = [(code_w[i:i + lines_per_page], []) for i in range(0, len(code_w), lines_per_page)]
    room = (height - 110 - len(pages[-1][0]) * line_h) // line_h # output goes on the last page if there is room
    if out_w and room < 4:
        pages.append((pages[-1][0][-3:], [])) # a new page, keep the last few lines of code for context
        room = (height - 110 - 3 * line_h) // line_h
    if len(out_w) > room:
        out_w = [[("...", COLORS["muted"])]] + out_w[-(room - 1):] # keep the end, usually the interesting part
    pages[-1] = (pages[-1][0], out_w)
    return size, pages

MIN_FONT = 18

def render_screen(screen, chapter, path, page=0):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (W, H), COLORS["bg"])
    draw = ImageDraw.Draw(img)
    if screen["kind"] in ("title", "chapter", "exercise"):
        if screen["kind"] == "exercise":
            draw.text((W // 2, 360), "Exercise: pause the video and try it yourself", font=font(54, bold=True), fill=COLORS["accent"], anchor="mm")
            body = textwrap.wrap(screen["text"], 70)
            for j, line in enumerate(body):
                draw.text((W // 2, 480 + 56 * j), line, font=font(38), fill=COLORS["text"], anchor="mm")
        else:
            subtitle = "nanochat post-training, from scratch" if screen["kind"] == "title" else screen.get("number", "")
            size = 64 if screen["kind"] == "title" else 76
            for j, line in enumerate(textwrap.wrap(screen["text"], 42)):
                draw.text((W // 2, 440 + (size + 20) * j), line, font=font(size, bold=True), fill=COLORS["text"], anchor="mm")
            draw.text((W // 2, 340), subtitle, font=font(36), fill=COLORS["muted"], anchor="mm")
        img.save(path)
        return
    size, pages = layout(screen)
    code_w, out_w = pages[page]
    fnt, line_h = font(size, mono=True), int(size * 1.4)
    # header
    draw.rectangle([0, 0, W, 72], fill=COLORS["header"])
    draw.text((40, 36), chapter, font=font(30, bold=True), fill=COLORS["text"], anchor="lm")
    label = screen.get("label", "") + (f"  ({page + 1}/{len(pages)})" if len(pages) > 1 else "")
    draw.text((W - 40, 36), label, font=font(26, mono=True), fill=COLORS["muted"], anchor="rm")
    left, right, y = 50, W - 50, 100
    code_bottom = y + len(code_w) * line_h + 40
    draw.rectangle([left, y, right, code_bottom], fill=COLORS["codebg"], outline=COLORS["border"])
    draw_lines(draw, code_w, left + 20, y + 20, fnt, line_h)
    if out_w:
        y = code_bottom + 30
        draw.rectangle([left, y, right, y + len(out_w) * line_h + 40], fill=COLORS["outbg"], outline=COLORS["border"])
        draw_lines(draw, out_w, left + 20, y + 20, fnt, line_h)
    img.save(path)

def build_video(blocks, tts, voice):
    outputs = notebook_outputs()
    frames_dir = os.path.join(BUILD, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    timeline = [] # dicts of: screen, audio path (or None for silence), seconds, subtitle text
    screen, chapter, chapter_number = None, "", 0
    for b in blocks:
        if b["type"] == "title":
            screen = {"kind": "title", "text": b["text"]}
        elif b["type"] == "chapter":
            chapter, chapter_number = b["text"], chapter_number + 1
            screen = {"kind": "chapter", "text": b["text"], "number": f"Chapter {chapter_number - 1}" if chapter_number > 1 else ""}
            timeline.append({"screen": screen, "chapter": chapter, "seconds": 1.5})
        elif b["type"] in ("code", "bash"):
            language = "python" if b["type"] == "code" else "bash"
            label = b.get("source") or ("notebook" if b["type"] == "code" else "terminal")
            output = outputs.get(b["code"], "") if b["type"] == "code" and b["exec"] else ""
            screen = {"kind": "code", "code": b["code"], "language": language, "label": label, "output": output}
            timeline.append({"screen": screen, "chapter": chapter, "seconds": 1.0})
        elif b["type"] == "output":
            screen = dict(screen, output=b["text"])
            timeline.append({"screen": screen, "chapter": chapter, "seconds": 1.5})
        elif b["type"] == "narration":
            timeline.append({"screen": screen, "chapter": chapter, "audio": audio_path(spoken(b["text"]), tts, voice), "text": b["text"]})
        elif b["type"] == "exercise":
            text = spoken("Okay, here's an exercise for you. " + b["text"] + " Pause the video here and give it a try.")
            card = {"kind": "exercise", "text": b["text"]}
            timeline.append({"screen": card, "chapter": chapter, "audio": audio_path(text, tts, voice),
                             "text": "Exercise: " + b["text"] + " Pause the video and give it a try."})
            timeline.append({"screen": card, "chapter": chapter, "seconds": 2.0})

    missing = [e["audio"] for e in timeline if e.get("audio") and not os.path.exists(e["audio"])]
    if missing:
        sys.exit(f"{len(missing)} narration segments have no audio yet, run the audio step first")
    for e in timeline:
        if e.get("audio"):
            e["seconds"] = duration(e["audio"])

    # video track: consecutive entries showing the same screen form a group, and if that screen has
    # several pages (long code), the pages are spread evenly over the time the group is on screen
    frames, frame_cache = [], {}
    def frame_for(screen, chapter, page):
        key = hashlib.sha1(json.dumps([screen, chapter, page], sort_keys=True).encode()).hexdigest()[:16]
        if key not in frame_cache:
            frame_cache[key] = os.path.join(frames_dir, f"{key}.png")
            render_screen(screen, chapter, frame_cache[key], page)
        return frame_cache[key]
    i = 0
    while i < len(timeline):
        j = i
        while j < len(timeline) and timeline[j]["screen"] == timeline[i]["screen"]:
            j += 1
        screen, chapter = timeline[i]["screen"], timeline[i]["chapter"]
        total = sum(e["seconds"] for e in timeline[i:j])
        num_pages = len(layout(screen)[1]) if screen["kind"] == "code" else 1
        for page in range(num_pages):
            frames.append((frame_for(screen, chapter, page), total / num_pages))
        i = j

    # audio track (narration + silences) and subtitles
    concat_audio, srt, t = [], [], 0.0
    silence = os.path.join(BUILD, "silence.mp3")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", "-t", "0.5",
                    "-b:a", "128k", "-ar", "44100", silence], check=True)
    for e in timeline:
        if e.get("audio"):
            concat_audio.append(e["audio"])
            srt.extend(subtitles(re.sub(r"[`*]", "", e["text"]), t, e["seconds"]))
        else:
            concat_audio.extend([silence] * round(e["seconds"] / 0.5))
        t += e["seconds"]
    with open(os.path.join(BUILD, "audio.txt"), "w") as f:
        f.writelines(f"file '{a}'\n" for a in concat_audio)
    with open(os.path.join(BUILD, "frames.txt"), "w") as f:
        for frame, seconds in frames:
            f.write(f"file '{frame}'\nduration {seconds:.3f}\n")
        f.write(f"file '{frames[-1][0]}'\n") # the concat demuxer needs the last frame twice
    with open(os.path.join(BUILD, "lecture.srt"), "w", encoding="utf-8") as f:
        f.write("\n".join(srt))
    out = os.path.join(BUILD, "lecture.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                    "-f", "concat", "-safe", "0", "-i", os.path.join(BUILD, "frames.txt"),
                    "-f", "concat", "-safe", "0", "-i", os.path.join(BUILD, "audio.txt"),
                    "-vf", "fps=10,format=yuv420p", "-c:v", "libx264", "-preset", "medium", "-tune", "stillimage",
                    "-c:a", "aac", "-b:a", "160k", "-shortest", out], check=True)
    print(f"wrote {out} ({t / 60:.1f} minutes) and lecture.srt")

def subtitles(text, start, seconds, max_chars=90):
    """ split a narration segment into subtitle cues, timed in proportion to their length """
    def stamp(x):
        h, m, s = int(x // 3600), int(x % 3600 // 60), x % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")
    sentences = re.split(r"(?<=[.?!])\s+", text)
    cues = []
    for sentence in sentences:
        cues.extend(textwrap.wrap(sentence, max_chars))
    total = sum(len(c) for c in cues)
    out, t = [], start
    for c in cues:
        dt = seconds * len(c) / total
        out.append(f"{len(out) + 1 + subtitles.counter}\n{stamp(t)} --> {stamp(t + dt)}\n{c}\n")
        t += dt
    subtitles.counter += len(out)
    return out
subtitles.counter = 0

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["notebook", "audio", "video", "all", "stats"])
    parser.add_argument("--execute", action="store_true", help="execute the notebook's python cells")
    parser.add_argument("--tts", choices=list(BACKENDS), default="kokoro", help="text-to-speech backend")
    parser.add_argument("--voice", default=None, help="voice for the backend (default: kokoro am_michael, ElevenLabs George)")
    parser.add_argument("--chapters", type=int, default=None, help="only the first N chapters (for a quick preview)")
    args = parser.parse_args()
    voice = args.voice or os.environ.get("ELEVENLABS_VOICE_ID" if args.tts == "elevenlabs" else "TTS_VOICE") or BACKENDS[args.tts][0]
    blocks = parse()
    if args.chapters is not None:
        chapter_starts = [i for i, b in enumerate(blocks) if b["type"] == "chapter"]
        if args.chapters < len(chapter_starts):
            blocks = blocks[:chapter_starts[args.chapters]]
    if args.step == "stats":
        segments = narration_segments(blocks)
        words = sum(len(s.split()) for s in segments)
        print(f"{len(segments)} segments, {words:,} words, {sum(len(s) for s in segments):,} characters, ~{words / 150:.0f} minutes")
    if args.step in ("notebook", "all"):
        build_notebook(blocks, execute=args.execute or args.step == "all")
    if args.step in ("audio", "all"):
        build_audio(blocks, args.tts, voice)
    if args.step in ("video", "all"):
        build_video(blocks, args.tts, voice)
