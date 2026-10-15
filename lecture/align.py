"""
Word timestamps for the narration, so that the on-screen actions (pointing, drawing, scrolling)
happen exactly when the corresponding word is spoken.

We transcribe each clip with Whisper (faster-whisper) and align its words to the script.
That works for any TTS backend. Without faster-whisper installed, we fall back to spreading
the words evenly over the clip, which is roughly right.
"""

import os
import re
import json
import difflib
import subprocess
import numpy as np

_model = None

def norm(word):
    return re.sub(r"[^a-z0-9]", "", word.lower())

def whisper_words(audio_path):
    """ [(word, start, end)] for an audio clip, cached next to it """
    cache = audio_path + ".words.json"
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)
    global _model
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return None
    if _model is None:
        _model = WhisperModel("small.en", device="cpu", compute_type="int8")
    pcm = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", audio_path, "-f", "s16le", "-ac", "1", "-ar", "16000", "-"],
                         capture_output=True, check=True).stdout
    audio = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768
    segments, _ = _model.transcribe(audio, beam_size=1, word_timestamps=True)
    words = [(w.word.strip(), w.start, w.end) for s in segments for w in s.words]
    with open(cache, "w") as f:
        json.dump(words, f)
    return words

def script_word_times(script_words, audio_path, seconds):
    """ The start time (in seconds, within the clip) of every word of the script """
    n = len(script_words)
    heard = whisper_words(audio_path)
    if not heard:
        return [seconds * i / max(n, 1) for i in range(n)]
    a = [norm(w) for w in script_words]
    b = [norm(w) for w, _, _ in heard]
    times = [None] * n
    for block in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_matching_blocks():
        for k in range(block.size):
            times[block.a + k] = heard[block.b + k][1]
    # words that didn't match (numbers, acronyms...): interpolate between their matched neighbours
    known = [i for i, t in enumerate(times) if t is not None]
    if not known:
        return [seconds * i / max(n, 1) for i in range(n)]
    for i in range(n):
        if times[i] is None:
            left = max((k for k in known if k < i), default=None)
            right = min((k for k in known if k > i), default=None)
            t_left = times[left] if left is not None else 0.0
            t_right = times[right] if right is not None else seconds
            i_left = left if left is not None else -1
            i_right = right if right is not None else n
            times[i] = t_left + (t_right - t_left) * (i - i_left) / (i_right - i_left)
    return times
