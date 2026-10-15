"""
The on-screen teacher: an original, clearly animated presenter (not a likeness of any real person),
lip-synced to the narration. The mouth follows the loudness and brightness of the audio, so it works
with any of the TTS backends. It blinks, raises its eyebrows on emphasis, and moves its head a little.

All the poses are rendered once as "sprites", and the video is a sequence of (sprite, duration).
"""

import os
import random
import subprocess
import numpy as np

SIZE = (400, 450) # the "webcam" panel, in pixels
S = 2 # draw at 2x and downsample, for smooth (antialiased) edges
MOUTHS = ["closed", "small", "medium", "wide", "round", "ee"]
EYES = ["open", "half", "closed"]
BROWS = ["neutral", "raised"]
TILTS = [-3, 0, 3] # degrees

SKIN, SKIN_SHADOW = "#e8b48f", "#d39b76"
HAIR = "#2f2a3a"
SHIRT, SHIRT_DARK = "#3d6ea8", "#2f5785"
BACKDROP_TOP, BACKDROP_BOTTOM = "#3a4252", "#262b36"

def _backdrop(w, h):
    from PIL import Image, ImageDraw
    top, bottom = np.array(Image.new("RGB", (1, 1), BACKDROP_TOP).getpixel((0, 0))), np.array(Image.new("RGB", (1, 1), BACKDROP_BOTTOM).getpixel((0, 0)))
    t = np.linspace(0, 1, h)[:, None, None]
    gradient = (top * (1 - t) + bottom * t).astype(np.uint8).repeat(w, axis=1)
    img = Image.fromarray(gradient, "RGB")
    draw = ImageDraw.Draw(img)
    # a soft light behind the head, and a little bookshelf, for that "recorded at home" feeling
    for r, c in [(260, "#46506a"), (220, "#4b5672"), (180, "#505b78")]:
        draw.ellipse([w // 2 - r * S // 2, 300 - r * S // 2, w // 2 + r * S // 2, 300 + r * S // 2], fill=c)
    shelf_y = 150
    draw.rectangle([20, shelf_y + 70, 200, shelf_y + 80], fill="#5a4636")
    for i, (bw, bh, c) in enumerate([(22, 64, "#c25e00"), (18, 56, "#2e7d32"), (26, 68, "#7b5ea7"), (20, 50, "#d0a030"), (24, 62, "#b03a48")]):
        x = 30 + sum(b[0] + 6 for b in [(22,), (18,), (26,), (20,), (24,)][:i])
        draw.rectangle([x, shelf_y + 70 - bh, x + bw, shelf_y + 70], fill=c)
    return img

def _head(mouth, eyes, brows):
    """ The head on a transparent layer, so that we can tilt it """
    from PIL import Image, ImageDraw
    w, h = SIZE[0] * S, SIZE[1] * S
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx, cy = w // 2, 330 # head center
    # ears, head, hair
    d.ellipse([cx - 175, cy - 20, cx - 125, cy + 60], fill=SKIN_SHADOW)
    d.ellipse([cx + 125, cy - 20, cx + 175, cy + 60], fill=SKIN_SHADOW)
    d.ellipse([cx - 150, cy - 175, cx + 150, cy + 185], fill=SKIN)
    d.chord([cx - 162, cy - 200, cx + 162, cy + 60], 180, 360, fill=HAIR)
    d.ellipse([cx - 165, cy - 120, cx - 110, cy + 10], fill=HAIR)
    d.ellipse([cx + 110, cy - 120, cx + 165, cy + 10], fill=HAIR)
    d.pieslice([cx - 60, cy - 215, cx + 190, cy - 30], 150, 330, fill=HAIR) # a swoosh of a fringe
    # eyes
    ey = cy + 15
    for ex in (cx - 62, cx + 62):
        if eyes == "closed":
            d.arc([ex - 30, ey - 14, ex + 30, ey + 14], 10, 170, fill="#3a2a22", width=6)
            continue
        d.ellipse([ex - 30, ey - 22, ex + 30, ey + 22], fill="white")
        d.ellipse([ex - 14, ey - 14, ex + 14, ey + 14], fill="#3b2c24")
        d.ellipse([ex - 4, ey - 10, ex + 4, ey - 2], fill="white") # a highlight makes them look alive
        if eyes == "half":
            d.rectangle([ex - 32, ey - 24, ex + 32, ey], fill=SKIN)
            d.line([ex - 30, ey, ex + 30, ey], fill="#3a2a22", width=5)
    # glasses
    for ex in (cx - 62, cx + 62):
        d.rounded_rectangle([ex - 48, ey - 38, ex + 48, ey + 36], radius=26, outline="#1e1e24", width=7)
    d.line([cx - 14, ey - 8, cx + 14, ey - 8], fill="#1e1e24", width=7)
    # eyebrows
    by = ey - 60 - (14 if brows == "raised" else 0)
    for ex in (cx - 62, cx + 62):
        d.line([ex - 34, by + 6, ex, by - 2, ex + 34, by + 6], fill=HAIR, width=12, joint="curve")
    # nose
    d.arc([cx - 18, cy + 50, cx + 18, cy + 90], 20, 160, fill=SKIN_SHADOW, width=7)
    # mouth
    my = cy + 125
    if mouth == "closed":
        d.arc([cx - 45, my - 25, cx + 45, my + 10], 20, 160, fill="#8a3b33", width=8)
    else:
        mw, mh = {"small": (34, 14), "medium": (42, 26), "wide": (50, 38), "round": (24, 26), "ee": (56, 16)}[mouth]
        d.ellipse([cx - mw, my - mh, cx + mw, my + mh], fill="#5b1f1f")
        if mouth in ("medium", "wide", "ee"):
            d.chord([cx - mw + 6, my - mh, cx + mw - 6, my - mh + max(10, mh // 2)], 180, 360, fill="white") # teeth
            d.rectangle([cx - mw + 10, my - mh + 2, cx + mw - 10, my - mh + max(6, mh // 3)], fill="white")
        if mouth in ("medium", "wide", "round"):
            d.chord([cx - mw // 2, my + mh // 3, cx + mw // 2, my + mh], 0, 180, fill="#c0504d") # tongue
    return img

def render_sprite(mouth, eyes, brows, tilt, path):
    from PIL import Image, ImageDraw, ImageFont
    w, h = SIZE[0] * S, SIZE[1] * S
    img = _backdrop(w, h).convert("RGBA")
    d = ImageDraw.Draw(img)
    cx = w // 2
    # body: shoulders in a t-shirt, and a neck
    d.rounded_rectangle([cx - 310, 590, cx + 310, h + 200], radius=170, fill=SHIRT)
    d.rectangle([cx - 58, 470, cx + 58, 610], fill=SKIN_SHADOW)
    d.chord([cx - 90, 550, cx + 90, 660], 0, 180, fill=SKIN_SHADOW)
    d.arc([cx - 100, 540, cx + 100, 680], 0, 180, fill=SHIRT_DARK, width=8) # collar
    head = _head(mouth, eyes, brows).rotate(tilt, center=(cx, 560), resample=Image.BICUBIC)
    img.alpha_composite(head)
    img = img.convert("RGB").resize(SIZE, Image.LANCZOS)
    # label: this is a synthetic presenter, say so
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    d.rounded_rectangle([10, SIZE[1] - 36, 160, SIZE[1] - 10], radius=8, fill=(0, 0, 0))
    d.text((20, SIZE[1] - 23), "AI narrator", font=font, fill="white", anchor="lm")
    img.save(path)

def sprite_path(sprites_dir, mouth, eyes, brows, tilt):
    path = os.path.join(sprites_dir, f"{mouth}_{eyes}_{brows}_{tilt}.png")
    if not os.path.exists(path):
        render_sprite(mouth, eyes, brows, tilt, path)
    return path

# -----------------------------------------------------------------------------
# Animating it from the audio

def load_audio(path, sr=16000):
    pcm = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-f", "s16le", "-ac", "1", "-ar", str(sr), "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(pcm, np.int16).astype(np.float32) / 32768

def mouth_shapes(audio, fps, sr=16000):
    """ One mouth shape per video frame: loudness picks how open, spectral brightness picks ee vs oo """
    hop = sr // fps
    n = len(audio) // hop
    frames = audio[:n * hop].reshape(n, hop)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-10)
    spectrum = np.abs(np.fft.rfft(frames * np.hanning(hop), axis=1))
    freqs = np.fft.rfftfreq(hop, 1 / sr)
    centroid = (spectrum * freqs).sum(axis=1) / (spectrum.sum(axis=1) + 1e-10)
    smooth = lambda x: np.convolve(x, np.ones(3) / 3, mode="same") # real mouths don't jitter every 40ms
    rms, centroid = smooth(rms), smooth(centroid)
    loud = rms / (np.percentile(rms, 95) + 1e-10) # normalize per clip
    shapes = []
    for level, c in zip(loud, centroid):
        if level < 0.12:
            shapes.append("closed")
        elif level < 0.35:
            shapes.append("small")
        elif c > 2600:
            shapes.append("ee")
        elif c < 900:
            shapes.append("round")
        elif level < 0.7:
            shapes.append("medium")
        else:
            shapes.append("wide")
    return shapes, loud

def build_track(segments, fps, sprites_dir, seed=1337):
    """
    segments: list of (audio path or None for silence, seconds), in order.
    Returns a list of (sprite path, seconds), run-length encoded.
    """
    os.makedirs(sprites_dir, exist_ok=True)
    rng = random.Random(seed)
    frames, t = [], 0.0 # (mouth, loudness) for every video frame of the whole lecture
    for audio, seconds in segments:
        n = round((t + seconds) * fps) - round(t * fps) # cumulative rounding, so we never drift out of sync
        t += seconds
        if audio is None:
            frames.extend([("closed", 0.0)] * n)
            continue
        shapes, loud = mouth_shapes(load_audio(audio), fps)
        shapes, loud = (shapes + ["closed"] * n)[:n], list(loud) + [0.0] * max(0, n - len(loud))
        frames.extend(zip(shapes, loud[:n]))
    track = []
    next_blink, blink_left = rng.uniform(2, 5), 0
    tilt, next_tilt = 0, rng.uniform(3, 8)
    brows_left, avg = 0, 0.0
    for i, (mouth, level) in enumerate(frames):
        t = i / fps
        # blink every few seconds: half, closed, half
        if t >= next_blink:
            blink_left, next_blink = 3, t + rng.uniform(2.5, 6.0)
        eyes = {3: "half", 2: "closed", 1: "half"}.get(blink_left, "open")
        blink_left = max(0, blink_left - 1)
        # raise the eyebrows on emphasis: a sudden jump in loudness
        avg = 0.95 * avg + 0.05 * level
        if level > 0.9 and level > 2.2 * avg and brows_left == 0 and rng.random() < 0.5:
            brows_left = int(0.5 * fps)
        brows = "raised" if brows_left > 0 else "neutral"
        brows_left = max(0, brows_left - 1)
        # tilt the head now and then, like people do when they talk
        if t >= next_tilt:
            tilt, next_tilt = rng.choice(TILTS), t + rng.uniform(3, 8)
        sprite = sprite_path(sprites_dir, mouth, eyes, brows, tilt)
        if track and track[-1][0] == sprite:
            track[-1] = (sprite, track[-1][1] + 1 / fps)
        else:
            track.append((sprite, 1 / fps))
    return track
