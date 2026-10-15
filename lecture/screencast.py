"""
Renders the lecture as a screencast, the way Karpathy records his: a Jupyter notebook where the code
gets typed out and run, an editor where we scroll through the real files and jump back and forth
between them, a terminal where training logs stream by, a whiteboard where diagrams get drawn, and a
mouse cursor that points at things and scribbles on the screen, timed to the words of the narration.

Direction lives inline in lecture.md, as cues inside the narration, placed right before the word
where they should happen (and always followed by a space):

  [[point "text"]]            move the mouse to some text on screen (scrolls there if needed)
  [[circle "text"]]           draw a circle around it
  [[underline "text"]]        underline it
  [[highlight "text"]]        highlighter marker over it
  [[note "text" "message"]]   a handwritten note next to it, with an arrow
  [[arrow "from" "to"]]       an arrow from one thing to another
  [[scroll "text"]]           scroll so that the text is near the top
  [[open FILE "text"]]        switch to the editor, open FILE (a tab) and scroll to the text
  [[notebook]] [[terminal]] [[board]]   switch to that view
  [[write "text"]]            write a line of handwriting on the whiteboard
  [[draw name]]               draw the whiteboard elements called @name
  [[clear]]                   erase the annotations

Targets are searched in the current view (the latest notebook cell first, then earlier ones).
"""

import os
import re
import math
import json
import shlex
import random
import hashlib
import subprocess
from multiprocessing import Pool

from PIL import Image, ImageDraw, ImageFont

import build as B
import align
import avatar

W, H = B.W, B.H
CW = B.CONTENT_W # content width, the sidebar is to the right of it
FPS = 25
TAB_H = 50
VIEW_H = H - TAB_H
CUE_RE = B.CUE_RE
INK = {"red": "#e0245e", "blue": "#1f6feb", "green": "#2e7d32", "orange": "#d97706", "black": "#222222",
       "gray": "#888888", "purple": "#7b4fd6"}
ANN = INK["red"]

strip_cues = B.strip_cues

# -----------------------------------------------------------------------------
# fonts and text layout

HAND_GLYPHS = {"→": "->", "←": "<-", "≈": "~", "≠": "!=", "²": "^2"} # missing from the handwriting font

def hand(text):
    for a, b in HAND_GLYPHS.items():
        text = text.replace(a, b)
    return text

_fonts = {}
def font(size, kind="mono", bold=False):
    key = (size, kind, bold)
    if key not in _fonts:
        if kind == "hand":
            path = os.path.join(B.BUILD, "fonts", "PatrickHand-Regular.ttf")
            try:
                _fonts[key] = ImageFont.truetype(path, size)
            except OSError:
                _fonts[key] = B.font(size)
        else:
            _fonts[key] = B.font(size, bold=bold, mono=kind == "mono")
    return _fonts[key]

CODE = font(21)
CHAR_W = CODE.getlength("x")
LINE_H = 30

class Doc:
    """ Monospace text, wrapped into rows, with a map from character offsets to (row, col) """

    def __init__(self, lines, max_cols):
        self.rows, self.row_of_line = [], []
        texts = []
        for line in lines:
            self.row_of_line.append(len(self.rows))
            row, n = [], 0
            for text, color in line:
                while n + len(text) > max_cols:
                    take = max_cols - n
                    row.append((text[:take], color))
                    self.rows.append(row)
                    row, n, text = [], 0, text[take:]
                row.append((text, color))
                n += len(text)
            self.rows.append(row)
            texts.append("".join(t for t, _ in line))
        self.max_cols = max_cols
        self.lines = texts
        self.text = "\n".join(texts)
        self.line_offsets = [0]
        for t in texts:
            self.line_offsets.append(self.line_offsets[-1] + len(t) + 1)

    def rowcol(self, offset):
        line = max(i for i, o in enumerate(self.line_offsets[:-1]) if o <= offset)
        col = offset - self.line_offsets[line]
        return self.row_of_line[line] + col // self.max_cols, col % self.max_cols

    def find(self, needle):
        i = self.text.find(needle)
        return None if i < 0 else (i, i + len(needle))

    def bbox(self, span, x0, y0):
        """ bounding box of the characters [a, b) when the doc is drawn at (x0, y0) """
        (r0, c0), (r1, c1) = self.rowcol(span[0]), self.rowcol(max(span[0], span[1] - 1))
        if r0 != r1:
            c0, c1 = 0, self.max_cols - 1
        return (x0 + c0 * CHAR_W, y0 + r0 * LINE_H, x0 + (c1 + 1) * CHAR_W, y0 + (r1 + 1) * LINE_H)

    def draw(self, draw, x0, y0, num_chars=None, y_min=-10**9, y_max=10**9):
        """ draw the rows (optionally only the first num_chars characters), returns the caret position """
        left, caret = num_chars, (x0, y0)
        for r, row in enumerate(self.rows):
            y = y0 + r * LINE_H
            x = x0
            for text, color in row:
                if left is not None:
                    if left <= 0:
                        return caret
                    text = text[:left]
                    left -= len(text)
                if y_min - LINE_H < y < y_max:
                    draw.text((x, y), text, font=CODE, fill=color)
                x += len(text) * CHAR_W
            caret = (x, y)
            if left is not None:
                left -= 1 # the newline at the end of the row (approximately, for wrapped rows)
        return caret

def code_lines(code, language="python"):
    return B.highlight(code, language) if code else [[]]

def text_lines(text):
    return B.ansi_lines(text)

# -----------------------------------------------------------------------------
# the views

class Cell:
    def __init__(self, index, code, output, number, heading=None):
        self.index, self.number, self.heading = index, number, heading
        width_cols = int((CW - 150 - 40) / CHAR_W)
        self.code = Doc(code_lines(code), width_cols) if heading is None else None
        self.out = Doc(text_lines(output), width_cols) if output else None
        self.num_chars = len(code)
        self.t_appear = self.t_type_end = self.t_run = 0.0

    def code_height(self, rows=None):
        if self.heading is not None:
            return 70
        rows = len(self.code.rows) if rows is None else rows
        return rows * LINE_H + 24

    def out_height(self):
        return len(self.out.rows) * LINE_H + 20 if self.out else 0

    def full_height(self):
        return self.code_height() + self.out_height()

    def chars_typed(self, t):
        if t >= self.t_type_end:
            return self.num_chars
        return int(self.num_chars * max(0.0, t - self.t_appear) / max(1e-6, self.t_type_end - self.t_appear))

    def height_at(self, t):
        if self.heading is not None:
            return self.code_height()
        k = self.chars_typed(t)
        rows = self.code.rowcol(k)[0] + 1 if k < self.num_chars else len(self.code.rows)
        return self.code_height(rows) + (self.out_height() if t >= self.t_run else 0)

CELL_GAP = 28
NB_X_CODE = 150

class Notebook:
    def __init__(self):
        self.cells = []

    def layout(self, t):
        """ [(cell, y)] of the cells that exist at time t, in content coordinates """
        y, out = 30, []
        for c in self.cells:
            if c.t_appear > t:
                break
            out.append((c, y))
            y += c.height_at(t) + CELL_GAP
        return out, y

    def cell_y(self, index):
        y = 30
        for c in self.cells[:index]:
            y += c.full_height() + CELL_GAP
        return y

    def find(self, needle, before_index):
        """ search the latest cell first (output, then code), then earlier cells """
        for c in reversed(self.cells[:before_index + 1]):
            if c.heading is not None:
                continue
            y = self.cell_y(c.index)
            for part, doc, x0, y0 in [("out", c.out, NB_X_CODE + 14, y + c.code_height() + 10), ("code", c.code, NB_X_CODE + 14, y + 12)]:
                if doc is None:
                    continue
                span = doc.find(needle)
                if span:
                    return doc.bbox(span, x0, y0), (c.t_run if part == "out" else c.t_appear)
        return None

class Editor:
    def __init__(self):
        self.files = {}

    def doc(self, filename):
        if filename not in self.files:
            with open(os.path.join(B.ROOT, filename), encoding="utf-8") as f:
                source = f.read().rstrip("\n")
            language = "python" if filename.endswith(".py") else "bash"
            self.files[filename] = Doc(code_lines(source, language), int((CW - 110 - 30) / CHAR_W))
        return self.files[filename]

    def line_y(self, filename, line):
        return 20 + self.doc(filename).row_of_line[line] * LINE_H

    def line_of(self, filename, needle):
        doc = self.doc(filename)
        i = doc.text.find(needle)
        return 0 if i < 0 else max(k for k, o in enumerate(doc.line_offsets[:-1]) if o <= i)

    def find(self, filename, needle, near_line=0):
        doc = self.doc(filename)
        # prefer the occurrence closest after the line we're looking at
        starts = [m.start() for m in re.finditer(re.escape(needle), doc.text)]
        if not starts:
            return None
        near = doc.line_offsets[min(near_line, len(doc.line_offsets) - 1)]
        start = min(starts, key=lambda s: (s < near, abs(s - near)))
        return doc.bbox((start, start + len(needle)), 100, 20)

class Terminal:
    def __init__(self):
        self.lines = [] # (text, kind, t_appear, t_typed) kind: cmd | out
        width_cols = int((CW - 60) / CHAR_W)
        self.width_cols = width_cols

    def doc(self):
        lines = []
        for text, kind, _, _ in self.lines:
            lines.append([("$ ", "#8ae234"), (text, "#ffffff")] if kind == "cmd" else [(text, "#d0d0d0")])
        return Doc(lines, self.width_cols)

class Board:
    def __init__(self):
        self.elements = [] # dicts: kind, args, name, t, dur
        self.next_write = 0

# -----------------------------------------------------------------------------
# hand-drawn marks

def wobble(seed):
    rng = random.Random(seed)
    return [rng.uniform(0, 2 * math.pi) for _ in range(3)]

def circle_points(bbox, seed, n=64):
    x0, y0, x1, y1 = bbox
    cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2 + 16, (y1 - y0) / 2 + 12
    p = wobble(seed)
    a0 = p[0]
    pts = []
    for i in range(n + 1):
        a = a0 + 2 * math.pi * 1.1 * i / n
        k = 1 + 0.05 * math.sin(3 * a + p[1]) + 0.03 * i / n
        pts.append((cx + rx * k * math.cos(a), cy + ry * k * math.sin(a)))
    return pts

def underline_points(bbox, seed, n=24):
    x0, _, x1, y1 = bbox
    p = wobble(seed)
    return [(x0 - 4 + (x1 - x0 + 8) * i / n, y1 + 3 + 2.5 * math.sin(p[0] + 6 * i / n)) for i in range(n + 1)]

def arrow_points(a, b, seed, n=30):
    (ax, ay), (bx, by) = a, b
    p = wobble(seed)
    mx, my = (ax + bx) / 2, (ay + by) / 2
    dx, dy = bx - ax, by - ay
    bend = 0.18 * (1 if p[0] > math.pi else -1)
    cx, cy = mx - dy * bend, my + dx * bend
    return [((1 - u) ** 2 * ax + 2 * (1 - u) * u * cx + u ** 2 * bx, (1 - u) ** 2 * ay + 2 * (1 - u) * u * cy + u ** 2 * by)
            for u in (i / n for i in range(n + 1))]

def draw_polyline(draw, pts, progress, color, width=4):
    k = max(2, int(len(pts) * progress))
    pts = pts[:k]
    if len(pts) >= 2:
        draw.line(pts, fill=color, width=width, joint="curve")
    return pts[-1]

def draw_arrowhead(draw, pts, color, width=4):
    (x1, y1), (x2, y2) = pts[-4], pts[-1]
    a = math.atan2(y2 - y1, x2 - x1)
    for s in (2.6, -2.6):
        draw.line([(x2, y2), (x2 + 18 * math.cos(a + s), y2 + 18 * math.sin(a + s))], fill=color, width=width)

def ease(u):
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)

# -----------------------------------------------------------------------------
# pass 1: the timeline of events

class Lecture:
    """ Everything that happens, with times. Built once from the blocks and the word timings. """

    def __init__(self, blocks, tts, voice):
        self.notebook, self.editor, self.terminal, self.board = Notebook(), Editor(), Terminal(), Board()
        self.events = [] # (t, kind, payload)
        self.chapters = [b["text"] for b in blocks if b["type"] == "chapter"]
        self.chapter_spans = [] # (name, t0, t1)
        self.audio = [] # (path or None, seconds, subtitle text)
        self.warnings = []
        outputs = B.notebook_outputs()
        t, chapter, number = 0.0, None, 0
        pending_output_for = None # the terminal command that the next output block belongs to
        screen_events = [] # times when the "screen" changes (for typing speed and progressive scrolling)
        blocks = list(blocks)
        for i, b in enumerate(blocks):
            kind = b["type"]
            if kind == "title":
                self.title = b["text"]
            elif kind == "chapter":
                if chapter is not None:
                    self.chapter_spans.append((chapter, chapter_t0, t))
                chapter, chapter_t0 = b["text"], t
                self.events.append((t, "card", {"kind": "title" if len(self.chapter_spans) == 0 else "chapter", "text": b["text"],
                                                "number": len(self.chapter_spans)}))
                self.audio.append((None, 1.5, None))
                t += 1.5
                self.events.append((t, "view", "notebook" if len(self.chapter_spans) else "board"))
                cell = Cell(len(self.notebook.cells), "", "", None, heading=b["text"])
                cell.t_appear = t
                self.notebook.cells.append(cell)
                self.events.append((t, "cell", cell))
            elif kind == "code" and b["exec"] and not b.get("source"):
                number += 1
                cell = Cell(len(self.notebook.cells), b["code"], outputs.get(b["code"], ""), number)
                cell.t_appear = t
                self.notebook.cells.append(cell)
                self.events.append((t, "view", "notebook"))
                self.events.append((t, "cell", cell))
                screen_events.append((t, cell))
                self.audio.append((None, 1.0, None))
                t += 1.0
            elif kind == "code":
                start, end = b["start"], b.get("end")
                doc_lines = self.editor.doc(b["source"]).lines
                line0 = next(k for k, line in enumerate(doc_lines) if start in line)
                line1 = next((k for k in range(line0 + 1, len(doc_lines)) if end is not None and end in doc_lines[k]), len(doc_lines))
                payload = {"file": b["source"], "line0": line0, "line1": line1}
                self.events.append((t, "view", "editor"))
                self.events.append((t, "excerpt", payload))
                screen_events.append((t, payload))
                self.audio.append((None, 1.0, None))
                t += 1.0
            elif kind == "bash":
                commands = [line for line in b["code"].split("\n") if line.strip()]
                payload = {"commands": commands, "t": t}
                self.events.append((t, "view", "terminal"))
                self.events.append((t, "command", payload))
                screen_events.append((t, payload))
                pending_output_for = payload
                self.audio.append((None, 1.0, None))
                t += 1.0
            elif kind == "output":
                self.events.append((t, "output", {"text": b["text"], "for": pending_output_for}))
                self.audio.append((None, 1.5, None))
                t += 1.5
            elif kind == "whiteboard":
                self.events.append((t, "board_block", b["elements"])) # named elements get drawn by [[draw name]] cues
            elif kind in ("narration", "exercise"):
                text = b["text"] if kind == "narration" else "Okay, here's an exercise for you. " + b["text"] + " Pause the video here and give it a try."
                said = B.spoken(strip_cues(text))
                path = B.audio_path(said, tts, voice)
                seconds = B.duration(path)
                if kind == "exercise":
                    self.events.append((t, "card", {"kind": "exercise", "text": strip_cues(b["text"])}))
                self.audio.append((path, seconds, "Exercise: " + strip_cues(b["text"]) + " Pause the video and give it a try." if kind == "exercise" else strip_cues(text)))
                self._cues(text, path, seconds, t)
                t += seconds
                if kind == "exercise":
                    self.audio.append((None, 2.0, None))
                    t += 2.0
                    self.events.append((t, "uncard", None))
        self.chapter_spans.append((chapter, chapter_t0, t))
        self.duration = t
        # typing speed: finish typing in the first part of the time the cell is on screen
        screen_times = [s for s, _ in screen_events] + [t]
        for (s, obj), s_next in zip(screen_events, screen_times[1:]):
            if isinstance(obj, Cell):
                available = max(1.0, s_next - s)
                seconds = max(obj.num_chars / 400.0, min(obj.num_chars / 40.0, 0.45 * available)) # 40 chars/s, faster if needed
                obj.t_type_end = s + max(0.6, seconds)
                obj.t_run = obj.t_type_end + 0.5
            elif "commands" in obj:
                obj["t_next"] = s_next
            else:
                obj["t_next"] = s_next
        self.events.sort(key=lambda e: e[0])

    def _cues(self, text, path, seconds, t0):
        """ turn the [[cues]] of a narration paragraph into timed events """
        pieces = CUE_RE.split(text) # text, cue, text, cue, ...
        script_words = B.spoken(strip_cues(text)).split()
        times = align.script_word_times(script_words, path, seconds)
        prefix = ""
        for k in range(1, len(pieces), 2):
            prefix += pieces[k - 1]
            word_index = len(B.spoken(prefix).split()) if prefix.strip() else 0
            word_t = times[min(word_index, len(times) - 1)] if times else 0.0
            try:
                args = shlex.split(pieces[k])
            except ValueError:
                self.warnings.append(f"bad cue: {pieces[k]}")
                continue
            self.events.append((t0 + word_t, "cue", args))

# -----------------------------------------------------------------------------
# pass 2: simulate, frame by frame: which view, scroll positions, the mouse, the annotations

class State:
    pass

editor_scroll_override = {} # file -> (time, scroll) set by cues

def simulate(lec):
    n = int(math.ceil(lec.duration * FPS))
    frames = [None] * n
    events = list(lec.events)
    ei = 0
    view, card = "board", None
    scroll = {"notebook": 0.0, "editor": {}, "terminal": 0.0}
    scroll_target = {"notebook": None, "terminal": None}
    editor_file, editor_excerpt, open_files = None, None, []
    used = {"terminal": False, "board": False}
    term_cmds = [] # payloads
    annotations, ann_id = [], 0
    cursor = [CW * 0.6, 600.0]
    cursor_target = None # (view, x, y content coords)
    strokes = [] # (t0, dur, view, points) the cursor follows the pen
    current_cell = None
    board_slot = 0
    for f in range(n):
        t = f / FPS
        while ei < len(events) and events[ei][0] <= t:
            et, kind, payload = events[ei]
            ei += 1
            if kind == "card":
                card = payload
                if payload["kind"] in ("chapter", "title"): # every chapter starts with a clean whiteboard
                    for el in lec.board.elements:
                        if el.get("t") is not None and "erased" not in el:
                            el["erased"] = et
                    board_slot = 0
            elif kind == "uncard":
                card = None
            elif kind == "view":
                if payload != view:
                    annotations = []
                view = payload
                if payload in used:
                    used[payload] = True
                if card and card["kind"] in ("chapter", "title"):
                    card = None
            elif kind == "cell":
                current_cell = payload
                card = None if (card and card["kind"] != "exercise") else card
                scroll_target["notebook"] = None # follow the bottom again
                annotations = []
                if payload.heading is None:
                    cursor_target = ("notebook", CW - 160, lec.notebook.cell_y(payload.index) + 20)
            elif kind == "excerpt":
                editor_file, editor_excerpt = payload["file"], payload
                if editor_file not in open_files:
                    open_files.append(editor_file)
                annotations = []
                card = None
                y = lec.editor.line_y(editor_file, payload["line0"])
                scroll["editor"].setdefault(editor_file, max(0.0, y - 90))
                payload["t0"] = et
                cursor_target = ("editor", 140, y + LINE_H / 2)
            elif kind == "command":
                used["terminal"] = True
                term_cmds.append(payload)
                payload["t0"] = et
                for c in payload["commands"]:
                    lec.terminal.lines.append((c, "cmd", et, et + 0.9))
                annotations = []
                card = None
            elif kind == "output":
                cmd = payload["for"]
                lines = payload["text"].split("\n")
                t_start = et + 0.3
                span = min(6.0, max(1.0, 0.04 * len(lines)))
                for j, line in enumerate(lines):
                    lec.terminal.lines.append((line, "out", t_start + span * j / max(1, len(lines)), None))
            elif kind == "board_block":
                used["board"] = True
                tt = et
                for el in payload:
                    el = dict(el)
                    el["t"], el["dur"] = (None, 0.7) if el.get("name") else (tt, 0.01)
                    lec.board.elements.append(el)
            elif kind == "cue":
                args = payload
                action = args[0]
                if action in ("notebook", "terminal", "board", "editor"):
                    view = action
                    annotations = []
                    if action == "board":
                        used["board"] = True
                    continue
                if action == "clear":
                    annotations = []
                    continue
                if action == "write":
                    used["board"] = True
                    if view != "board":
                        view, annotations = "board", []
                    el = {"kind": "write", "args": [args[1]], "name": None, "t": et, "dur": max(0.6, 0.05 * len(args[1])), "slot": board_slot}
                    board_slot += 1
                    lec.board.elements.append(el)
                    strokes.append((et, el["dur"], "board", [(70, 120 + 70 * el["slot"]), (70 + 17 * len(args[1]), 120 + 70 * el["slot"])]))
                    continue
                if action == "erase":
                    for el in lec.board.elements:
                        if el.get("t") is not None and el["t"] <= et and "erased" not in el:
                            el["erased"] = et
                    board_slot = 0
                    continue
                if action == "draw":
                    tt = et
                    for el in lec.board.elements:
                        if el.get("name") == args[1] and el.get("t") is None:
                            el["t"] = tt
                            tt += 0.5
                            box = board_bbox(el)
                            strokes.append((el["t"], el["dur"], "board", [(box[0], box[1]), (box[2], box[3])]))
                    if view != "board":
                        view, annotations = "board", []
                    continue
                if action == "open":
                    filename = args[1]
                    if filename not in open_files:
                        open_files.append(filename)
                    editor_file, view, annotations = filename, "editor", []
                    box = lec.editor.find(filename, args[2]) if len(args) > 2 else (100, 20, 200, 50)
                    if box is None:
                        lec.warnings.append(f"open: '{args[2]}' not found in {filename}")
                        continue
                    editor_scroll_override[filename] = (et, max(0.0, box[1] - 120))
                    editor_excerpt = {"file": filename, "line0": None, "line1": None, "t0": et, "t_next": et,
                                      "near": lec.editor.line_of(filename, args[2])}
                    cursor_target = ("editor", box[0] + 10, (box[1] + box[3]) / 2)
                    continue
                # everything else targets some text on the screen
                targets = []
                for needle in args[1:3 if action == "arrow" else 2]:
                    box, t_ready = locate(lec, view, needle, current_cell, editor_file, editor_excerpt)
                    if box is None:
                        lec.warnings.append(f"[{et:.1f}s, {view}] {action}: could not find {needle!r}")
                        break
                    targets.append((box, t_ready))
                if len(targets) < (2 if action == "arrow" else 1):
                    continue
                box = targets[0][0]
                start = max(et, max(tr for _, tr in targets) + 0.1)
                # scroll there if it isn't visible
                s = current_scroll(scroll, view, editor_file)
                if box[1] < s + 60 or box[3] > s + VIEW_H - 80 or action == "scroll":
                    new_s = max(0.0, box[1] - (60 if action == "scroll" else VIEW_H * 0.3))
                    if view == "editor":
                        scroll["editor"][editor_file] = scroll["editor"].get(editor_file, 0.0) # keep, set target below
                        editor_scroll_override[editor_file] = (et, new_s)
                    elif view in scroll_target:
                        scroll_target[view] = new_s
                if action == "scroll":
                    continue
                cx, cy = box[0] + 6, (box[1] + box[3]) / 2
                cursor_target = (view, cx, cy)
                if action == "point":
                    continue
                ann_id += 1
                ann = {"view": view, "kind": action, "box": box, "t": start + 0.15, "seed": ann_id}
                if action == "note":
                    ann["text"] = args[2]
                    ann["dur"] = 0.4 + 0.04 * len(args[2])
                elif action == "arrow":
                    ann["box2"] = targets[1][0]
                    ann["dur"] = 0.6
                else:
                    ann["dur"] = 0.55
                annotations.append(ann)
                pts = annotation_path(ann)
                strokes.append((ann["t"], ann["dur"], view, pts))
        # scrolling
        if True:
            cells, bottom = lec.notebook.layout(t)
            follow = max(0.0, bottom - VIEW_H + 40)
            target = scroll_target["notebook"] if scroll_target["notebook"] is not None else follow
            scroll["notebook"] += (target - scroll["notebook"]) * min(1.0, 6.0 / FPS)
        if editor_file is not None:
            target = editor_scroll(lec, editor_file, editor_excerpt, t, scroll)
            cur = scroll["editor"].get(editor_file, target)
            scroll["editor"][editor_file] = cur + (target - cur) * min(1.0, 6.0 / FPS)
        visible = [ln for ln in lec.terminal.lines if ln[2] <= t]
        term_target = max(0.0, (len(visible) + 1) * LINE_H - VIEW_H + 60)
        scroll["terminal"] += (term_target - scroll["terminal"]) * min(1.0, 8.0 / FPS)
        # the mouse: follow the pen while drawing, otherwise glide to the target
        pen = None
        for (t0, dur, v, pts) in strokes:
            if t0 <= t <= t0 + dur and v == view:
                u = (t - t0) / dur
                pen = pts[min(len(pts) - 1, int(u * (len(pts) - 1)))]
        s = current_scroll(scroll, view, editor_file)
        if pen is not None:
            goal = (pen[0], pen[1] - (0 if view == "board" else s) + TAB_H)
            cursor[0] += (goal[0] - cursor[0]) * 0.6
            cursor[1] += (goal[1] - cursor[1]) * 0.6
        elif cursor_target is not None and cursor_target[0] == view:
            goal = (cursor_target[1], cursor_target[2] - (0 if view == "board" else s) + TAB_H)
            cursor[0] += (goal[0] - cursor[0]) * min(1.0, 9.0 / FPS)
            cursor[1] += (goal[1] - cursor[1]) * min(1.0, 9.0 / FPS)
        st = State()
        st.t, st.view, st.card = t, view, card
        st.nb_scroll, st.term_scroll = scroll["notebook"], scroll["terminal"]
        st.editor_file = editor_file
        st.editor_scroll = scroll["editor"].get(editor_file, 0.0) if editor_file else 0.0
        st.excerpt = editor_excerpt
        st.open_files = list(open_files)
        st.used = dict(used)
        annotations = [a for a in annotations if t - a["t"] < 14.0]
        st.annotations = [a for a in annotations if a["t"] <= t]
        st.cursor = (round(cursor[0]), round(cursor[1]))
        frames[f] = st
    return frames

def current_scroll(scroll, view, editor_file):
    if view == "notebook":
        return scroll["notebook"]
    if view == "editor":
        return scroll["editor"].get(editor_file, 0.0)
    if view == "terminal":
        return scroll["terminal"]
    return 0.0

def editor_scroll(lec, filename, excerpt, t, scroll):
    """ where the editor should be scrolled to: an explicit target, or slowly through a long excerpt """
    override = editor_scroll_override.get(filename)
    if override is not None and (excerpt is None or override[0] >= excerpt.get("t0", 0)):
        return override[1]
    if excerpt is None or excerpt.get("line0") is None or excerpt["file"] != filename:
        return scroll["editor"].get(filename, 0.0)
    y0 = lec.editor.line_y(filename, excerpt["line0"])
    y1 = lec.editor.line_y(filename, excerpt["line1"]) if excerpt["line1"] < len(lec.editor.doc(filename).lines) else lec.editor.line_y(filename, len(lec.editor.doc(filename).lines) - 1) + LINE_H
    start = max(0.0, y0 - 90)
    end = max(start, y1 - VIEW_H + 120)
    # read through the excerpt: start scrolling after a few seconds, arrive near the end of its time
    t0, t1 = excerpt["t0"] + 4.0, excerpt.get("t_next", excerpt["t0"]) - 3.0
    if end <= start or t1 <= t0:
        return start
    return start + (end - start) * ease((t - t0) / (t1 - t0))

def locate(lec, view, needle, current_cell, editor_file, excerpt):
    """ (bbox in content coordinates, time when it's on screen) """
    if view == "notebook":
        found = lec.notebook.find(needle, current_cell.index if current_cell else len(lec.notebook.cells) - 1)
        return found if found else (None, 0)
    if view == "editor" and editor_file:
        near = (excerpt["line0"] if excerpt.get("line0") is not None else excerpt.get("near", 0)) if excerpt and excerpt["file"] == editor_file else 0
        return lec.editor.find(editor_file, needle, near), 0
    if view == "terminal":
        doc = lec.terminal.doc()
        starts = [m.start() for m in re.finditer(re.escape(needle), doc.text)]
        if not starts:
            return None, 0
        s = starts[-1]
        line = max(i for i, o in enumerate(doc.line_offsets[:-1]) if o <= s)
        return doc.bbox((s, s + len(needle)), 30, 20), lec.terminal.lines[line][2]
    if view == "board":
        if needle.startswith("@"):
            boxes = [board_bbox(el) for el in lec.board.elements if el.get("name") == needle[1:]]
            if boxes:
                return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)), 0
        for el in lec.board.elements:
            if el["kind"] in ("text", "write", "box", "fill") and needle in " ".join(el["args"]):
                return board_bbox(el), 0
    return None, 0

def annotation_path(ann):
    if ann["kind"] == "circle":
        return circle_points(ann["box"], ann["seed"])
    if ann["kind"] == "underline":
        return underline_points(ann["box"], ann["seed"])
    if ann["kind"] == "highlight":
        x0, y0, x1, y1 = ann["box"]
        return [(x0 + (x1 - x0) * i / 10, (y0 + y1) / 2) for i in range(11)]
    if ann["kind"] == "arrow":
        a, b = ann["box"], ann["box2"]
        return arrow_points(((a[0] + a[2]) / 2, a[3] + 6), (b[0] - 8, (b[1] + b[3]) / 2), ann["seed"])
    if ann["kind"] == "note":
        x0, y0, x1, y1 = ann["box"]
        nx, ny = note_anchor(ann)
        return [(nx, ny), (nx + 15 * len(ann["text"]), ny)]
    return [(ann["box"][0], ann["box"][1])]

def note_anchor(ann):
    """ where the note goes: above the target (or below, near the top of the screen), shifted right a bit """
    x0, y0, x1, y1 = ann["box"]
    width = 17 * len(ann["text"])
    x = max(40, min(x0 + 60, CW - width - 50))
    return (x, y0 - 62) if y0 - 62 > 20 else (x, y1 + 18)

# -----------------------------------------------------------------------------
# whiteboard elements

def board_bbox(el):
    k, a = el["kind"], el["args"]
    if k == "write":
        y = 120 + 70 * el["slot"]
        return (60, y - 30, 70 + 17 * len(a[0]), y + 30)
    nums = [float(x) for x in a if re.fullmatch(r"-?\d+(\.\d+)?", x)]
    if k in ("box", "fill"):
        x, y, w, h = nums[:4]
        return (x, y, x + w, y + h)
    if k in ("arrow", "line"):
        x1, y1, x2, y2 = nums[:4]
        return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
    if k == "circle":
        cx, cy, r = nums[:3]
        return (cx - r, cy - r, cx + r, cy + r)
    if k == "text": # text X Y "string" [size] [color]
        x, y, text = float(a[0]), float(a[1]), a[2]
        size = float(a[3]) if len(a) > 3 and a[3] not in INK else 40
        return (x, y, x + 0.45 * size * len(text), y + size)
    return (0, 0, 10, 10)

def draw_board_element(draw, el, u):
    """ draw a whiteboard element, u in [0, 1] is how far along the drawing is """
    k, a = el["kind"], el["args"]
    color = next((INK[x] for x in a if x in INK), INK["black"])
    texts = [x for x in a if not re.fullmatch(r"-?\d+(\.\d+)?", x) and x not in INK]
    nums = [float(x) for x in a if re.fullmatch(r"-?\d+(\.\d+)?", x)]
    seed = hash(json.dumps(el.get("args"))) % 1000
    if k == "write":
        y = 120 + 70 * el["slot"]
        text = hand(a[0])[:max(1, int(len(hand(a[0])) * u))]
        draw.text((60, y), text, font=font(46, "hand"), fill=INK["black"], anchor="lm")
    elif k == "text":
        x, y, full = float(a[0]), float(a[1]), hand(a[2])
        size = int(float(a[3])) if len(a) > 3 and a[3] not in INK else 40
        draw.text((x, y), full[:max(1, int(len(full) * u))], font=font(size, "hand"), fill=color)
    elif k in ("box", "fill"):
        x, y, w, h = nums[:4]
        if k == "fill":
            fill = {"red": "#fde2e4", "green": "#dcf3df", "blue": "#dde9fb", "orange": "#fdebd3", "gray": "#eeeeee",
                    "purple": "#ebe2fb", "black": "#e8e8e8"}[next((x for x in a if x in INK), "gray")]
            if u > 0.3:
                draw.rounded_rectangle([x, y, x + w, y + h], radius=10, fill=fill)
        pts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)]
        perimeter = []
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            perimeter += [(x1 + (x2 - x1) * i / 12, y1 + (y2 - y1) * i / 12) for i in range(12)]
        draw_polyline(draw, perimeter, u, color, width=4)
        if texts and u > 0.6:
            size = int(nums[4]) if len(nums) > 4 else 34
            draw.text((x + w / 2, y + h / 2), hand(texts[0]), font=font(size, "hand"), fill=INK["black"], anchor="mm")
    elif k in ("arrow", "line"):
        x1, y1, x2, y2 = nums[:4]
        pts = [(x1 + (x2 - x1) * i / 20, y1 + (y2 - y1) * i / 20) for i in range(21)]
        draw_polyline(draw, pts, u, color, width=4)
        if k == "arrow" and u >= 1.0:
            draw_arrowhead(draw, pts, color)
    elif k == "circle":
        cx, cy, r = nums[:3]
        draw_polyline(draw, circle_points((cx - r + 16, cy - r + 12, cx + r - 16, cy + r - 12), seed), u, color)

# -----------------------------------------------------------------------------
# pass 3: drawing the frames

PROMPT = font(19)
TAB_FONT = font(19, "sans")

def draw_tabs(img, st):
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, CW, TAB_H], fill="#e9eaee")
    tabs = [("posttraining.ipynb", "notebook")] + [(f, ("editor", f)) for f in st.open_files]
    if st.used.get("terminal"):
        tabs.append(("terminal", "terminal"))
    if st.used.get("board"):
        tabs.append(("whiteboard", "board"))
    x = 12
    for label, key in tabs:
        active = (key == st.view) or (isinstance(key, tuple) and st.view == "editor" and key[1] == st.editor_file)
        w = TAB_FONT.getlength(label) + 36
        if x + w > CW - 10:
            break
        d.rounded_rectangle([x, 8, x + w, TAB_H + 6], radius=8, fill="#ffffff" if active else "#dcdde3")
        d.text((x + 18, 29), label, font=TAB_FONT, fill="#1f1f1f" if active else "#666666", anchor="lm")
        x += w + 6

_cell_cache = {}
def cell_image(cell, k, ran, running):
    key = (cell.index, k, ran, running)
    if key in _cell_cache:
        return _cell_cache[key]
    if len(_cell_cache) > 400:
        _cell_cache.clear()
    if cell.heading is not None:
        img = Image.new("RGB", (CW, cell.code_height()), "#ffffff")
        d = ImageDraw.Draw(img)
        d.text((NB_X_CODE, 12), cell.heading, font=font(40, "sans", bold=True), fill="#1f1f1f")
        _cell_cache[key] = img
        return img
    rows = cell.code.rowcol(k)[0] + 1 if k < cell.num_chars else len(cell.code.rows)
    code_h = cell.code_height(rows)
    h = code_h + (cell.out_height() if ran else 0)
    img = Image.new("RGB", (CW, h), "#ffffff")
    d = ImageDraw.Draw(img)
    label = f"[{cell.number}]:" if ran else ("[*]:" if running else "[ ]:")
    d.text((NB_X_CODE - 12, 12), label, font=PROMPT, fill="#303f9f", anchor="ra")
    d.rectangle([NB_X_CODE, 0, CW - 30, code_h - 4], fill="#f5f6f8", outline="#d8dbe0")
    caret = cell.code.draw(d, NB_X_CODE + 14, 12, num_chars=k if k < cell.num_chars else None)
    if k < cell.num_chars:
        d.line([caret[0] + 1, caret[1] + 3, caret[0] + 1, caret[1] + LINE_H - 4], fill="#1f1f1f", width=2)
    if ran and cell.out:
        cell.out.draw(d, NB_X_CODE + 14, code_h + 10)
    _cell_cache[key] = img
    return img

_editor_cache = {}
def editor_image(lec, filename):
    if filename not in _editor_cache:
        doc = lec.editor.doc(filename)
        img = Image.new("RGB", (CW, len(doc.rows) * LINE_H + 40 + VIEW_H), "#ffffff")
        d = ImageDraw.Draw(img)
        for line, row in enumerate(doc.row_of_line):
            d.text((80, 20 + row * LINE_H), str(line + 1), font=CODE, fill="#a0a4ad", anchor="ra")
        doc.draw(d, 100, 20)
        _editor_cache[filename] = img
    return _editor_cache[filename]

def render_card(card):
    img = Image.new("RGB", (CW, H), "#ffffff")
    d = ImageDraw.Draw(img)
    if card["kind"] == "exercise":
        d.text((CW // 2, 330), "Exercise: pause the video and try it yourself", font=font(48, "sans", True), fill=B.COLORS["accent"], anchor="mm")
        import textwrap
        for j, line in enumerate(textwrap.wrap(card["text"], 52)):
            d.text((CW // 2, 450 + 60 * j), line, font=font(44, "hand"), fill="#1f1f1f", anchor="mm")
    else:
        import textwrap
        sub = "nanochat post-training, from scratch" if card["kind"] == "title" else f"Chapter {card['number']}"
        text = card["text"] if card["kind"] != "title" else card.get("title", card["text"])
        d.text((CW // 2, 340), sub, font=font(36, "sans"), fill="#7a7a7a", anchor="mm")
        for j, line in enumerate(textwrap.wrap(text, 30)):
            d.text((CW // 2, 450 + 90 * j), line, font=font(68, "sans", True), fill="#1f1f1f", anchor="mm")
    return img

def draw_annotations(img, st, scroll):
    d = ImageDraw.Draw(img)
    for ann in st.annotations:
        u = min(1.0, (st.t - ann["t"]) / ann["dur"])
        dy = TAB_H - scroll
        shift = lambda pts: [(x, y + dy) for x, y in pts]
        if ann["kind"] == "highlight":
            x0, y0, x1, y1 = ann["box"]
            x1 = x0 + (x1 - x0) * u
            box = [int(x0 - 3), int(y0 + dy + 2), int(x1 + 3), int(y1 + dy - 2)]
            if box[2] > box[0] and box[3] > TAB_H:
                region = img.crop(box)
                img.paste(Image.blend(region, Image.new("RGB", region.size, "#ffe45c"), 0.45), box[:2])
        elif ann["kind"] == "note":
            nx, ny = note_anchor(ann)
            full = hand(ann["text"])
            text = full[:max(1, int(len(full) * min(1.0, u * 1.3)))]
            f = font(36, "hand")
            d.rounded_rectangle([nx - 10, ny + dy - 2, nx + f.getlength(full) + 10, ny + dy + 44], radius=10,
                                fill="#fffdf5", outline="#f2c4cf", width=2)
            d.text((nx, ny + dy), text, font=f, fill=ANN)
            if u > 0.75:
                x0, y0, x1, y1 = ann["box"]
                above = ny < y0
                start = (nx + 20, ny + dy + (44 if above else -2))
                end = (min(max(nx + 20, x0 + 10), x1 - 4), (y0 if above else y1) + dy + (-2 if above else 2))
                pts = arrow_points(start, end, ann["seed"], n=12)
                draw_polyline(d, pts, 1.0, ANN, width=3)
                draw_arrowhead(d, pts, ANN, width=3)
        else:
            pts = shift(annotation_path(ann))
            draw_polyline(d, pts, u, ANN, width=4)
            if ann["kind"] == "arrow" and u >= 1.0:
                draw_arrowhead(d, pts, ANN)

CURSOR = None
def cursor_sprite():
    global CURSOR
    if CURSOR is None:
        s = 2
        img = Image.new("RGBA", (24 * s, 36 * s), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        pts = [(0, 0), (0, 26), (6, 20), (11, 32), (15, 30), (10, 19), (18, 19)]
        d.polygon([(x * s + 2, y * s + 2) for x, y in pts], fill="black", outline="white", width=3)
        CURSOR = img.resize((24, 36), Image.LANCZOS)
    return CURSOR

def render_frame(lec, st, chrome):
    img = chrome.copy()
    if st.card is not None:
        img.paste(card_image(st.card), (0, 0))
        return img
    content = Image.new("RGB", (CW, H), "#ffffff")
    scroll = 0.0
    if st.view == "notebook":
        scroll = st.nb_scroll
        cells, _ = lec.notebook.layout(st.t)
        for cell, y in cells:
            top = y - scroll + TAB_H
            h = cell.height_at(st.t)
            if top + h < TAB_H or top > H:
                continue
            k = cell.chars_typed(st.t) if cell.heading is None else 0
            ran = st.t >= cell.t_run and cell.heading is None
            running = cell.t_type_end <= st.t < cell.t_run
            content.paste(cell_image(cell, k, ran, running), (0, int(top)))
    elif st.view == "editor" and st.editor_file:
        scroll = st.editor_scroll
        full = editor_image(lec, st.editor_file)
        crop = full.crop((0, int(scroll), CW, int(scroll) + VIEW_H))
        ex = st.excerpt
        if ex and ex.get("line0") is not None and ex["file"] == st.editor_file:
            d = ImageDraw.Draw(crop)
            y0 = lec.editor.line_y(st.editor_file, ex["line0"]) - scroll
            y1 = lec.editor.line_y(st.editor_file, ex["line1"]) - scroll if ex["line1"] < len(lec.editor.doc(st.editor_file).lines) else y0 + 10**5
            d.rectangle([92, max(0, y0 - 4), 96, min(VIEW_H, y1 - 4)], fill="#f0b429") # a bar marking the part we're talking about
        content.paste(crop, (0, TAB_H))
    elif st.view == "terminal":
        scroll = st.term_scroll
        content.paste(terminal_image(lec, st), (0, TAB_H))
    elif st.view == "board":
        content.paste(board_image(lec, st), (0, TAB_H))
    draw_tabs(content, st)
    draw_annotations(content, st, scroll if st.view != "board" else 0.0)
    img.paste(content.crop((0, 0, CW, H)), (0, 0))
    if 0 <= st.cursor[0] < CW - 10 and TAB_H <= st.cursor[1] < H - 10:
        sprite = cursor_sprite()
        img.paste(sprite, (int(st.cursor[0]), int(st.cursor[1])), sprite)
    return img

_card_cache = {}
def card_image(card):
    key = json.dumps(card, sort_keys=True)
    if key not in _card_cache:
        _card_cache[key] = render_card(card)
    return _card_cache[key]

def terminal_image(lec, st):
    img = Image.new("RGB", (CW, VIEW_H), "#1e1f22")
    d = ImageDraw.Draw(img)
    lines = [ln for ln in lec.terminal.lines if ln[2] <= st.t]
    y = 20 - st.term_scroll
    for text, kind, t_appear, t_typed in lines:
        if kind == "cmd":
            shown = text if st.t >= t_typed else text[:int(len(text) * (st.t - t_appear) / (t_typed - t_appear))]
            segs = [("$ ", "#8ae234"), (shown, "#ffffff")]
        else:
            segs = [(text, "#d0d0d0")]
        rows = Doc([segs], lec.terminal.width_cols).rows
        for row in rows:
            if -LINE_H < y < VIEW_H:
                x = 30
                for s, c in row:
                    d.text((x, y), s, font=CODE, fill=c)
                    x += len(s) * CHAR_W
            y += LINE_H
    return img

def board_image(lec, st):
    img = Image.new("RGB", (CW, VIEW_H), "#fcfcf8")
    d = ImageDraw.Draw(img)
    for el in lec.board.elements:
        if el.get("t") is None or el["t"] > st.t or ("erased" in el and el["erased"] <= st.t):
            continue
        draw_board_element(d, el, min(1.0, (st.t - el["t"]) / el["dur"]))
    return img

def frame_key(lec, st):
    """ two frames with the same key look the same, so we can reuse the bytes """
    ann = tuple((a["seed"], min(1.0, round((st.t - a["t"]) / a["dur"], 2))) for a in st.annotations)
    if st.card is not None:
        return ("card", json.dumps(st.card, sort_keys=True))
    if st.view == "notebook":
        cells, _ = lec.notebook.layout(st.t)
        c = cells[-1][0] if cells else None
        dyn = (len(cells), c.chars_typed(st.t) if c and c.heading is None else 0, st.t >= c.t_run if c else 0,
               c.t_type_end <= st.t < c.t_run if c else 0, round(st.nb_scroll))
    elif st.view == "editor":
        dyn = (st.editor_file, round(st.editor_scroll), json.dumps(st.excerpt and {k: v for k, v in st.excerpt.items() if k in ("file", "line0", "line1")}))
    elif st.view == "terminal":
        lines = [ln for ln in lec.terminal.lines if ln[2] <= st.t]
        last = lines[-1] if lines else None
        typing = int((st.t - last[2]) * 30) if last and last[1] == "cmd" and st.t < last[3] else -1
        dyn = (len(lines), typing, round(st.term_scroll))
    else:
        dyn = tuple(min(1.0, round((st.t - el["t"]) / el["dur"], 2)) for el in lec.board.elements
                    if el.get("t") is not None and el["t"] <= st.t and not ("erased" in el and el["erased"] <= st.t))
    return (st.view, dyn, ann, st.cursor, tuple(st.open_files), tuple(sorted(st.used.items())))

# -----------------------------------------------------------------------------
# putting it all together

_lec, _frames, _chrome = None, None, {}

def chrome_for(chapter):
    if chapter not in _chrome:
        img = Image.new("RGB", (W, H), "#ffffff")
        d = ImageDraw.Draw(img)
        B.draw_sidebar(d, chapter)
        _chrome[chapter] = img
    return _chrome[chapter]

def render_chunk(job):
    c, name, f0, f1, teacher_txt, out = job
    lec, frames = _lec, _frames
    chrome = chrome_for(name)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-"]
    if teacher_txt:
        cmd += ["-f", "concat", "-safe", "0", "-i", teacher_txt, "-filter_complex",
                f"[1:v]fps={FPS}[t];[0:v][t]overlay={B.AVATAR_XY[0]}:{B.AVATAR_XY[1]},format=yuv420p[v]", "-map", "[v]"]
    else:
        cmd += ["-vf", "format=yuv420p"]
    cmd += ["-frames:v", str(f1 - f0), "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    last_key, last_bytes = None, None
    for f in range(f0, f1):
        st = frames[min(f, len(frames) - 1)]
        key = frame_key(lec, st)
        if key != last_key:
            last_bytes = render_frame(lec, st, chrome).tobytes()
            last_key = key
        proc.stdin.write(last_bytes)
    proc.stdin.close()
    proc.wait()
    return out

def render(blocks, tts, voice, teacher=True, workers=3):
    global _lec, _frames
    B.CHAPTERS[:] = [b["text"] for b in blocks if b["type"] == "chapter"]
    lec = Lecture(blocks, tts, voice)
    for b in blocks:
        if b["type"] == "title":
            for t, kind, payload in lec.events:
                if kind == "card" and payload["kind"] == "title":
                    payload["title"] = b["text"]
    frames = simulate(lec)
    for w in lec.warnings:
        print("WARNING:", w)
    _lec, _frames = lec, frames
    # audio track and subtitles
    os.makedirs(B.BUILD, exist_ok=True)
    silence = os.path.join(B.BUILD, "silence.mp3")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", "-t", "0.5",
                    "-b:a", "128k", "-ar", "44100", silence], check=True)
    concat_audio, srt, t = [], [], 0.0
    B.subtitles.counter = 0
    for path, seconds, text in lec.audio:
        if path:
            concat_audio.append(path)
            srt.extend(B.subtitles(text, t, seconds))
        else:
            concat_audio.extend([silence] * round(seconds / 0.5))
        t += seconds
    with open(os.path.join(B.BUILD, "audio.txt"), "w") as f:
        f.writelines(f"file '{a}'\n" for a in concat_audio)
    with open(os.path.join(B.BUILD, "lecture.srt"), "w", encoding="utf-8") as f:
        f.write("\n".join(srt))
    # one chunk per chapter, rendered in parallel
    chunks_dir = os.path.join(B.BUILD, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)
    jobs = []
    for c, (name, t0, t1) in enumerate(lec.chapter_spans):
        f0, f1 = round(t0 * FPS), round(t1 * FPS)
        teacher_txt = None
        if teacher:
            segments, tt = [], 0.0
            for path, seconds, _ in lec.audio:
                if tt + 1e-6 >= t0 and tt < t1 - 1e-6:
                    segments.append((path, seconds))
                tt += seconds
            track = avatar.build_track(segments, FPS, os.path.join(B.BUILD, "sprites"), seed=c)
            teacher_txt = os.path.join(chunks_dir, f"teacher_{c:02d}.txt")
            with open(teacher_txt, "w") as f:
                for sprite, seconds in track:
                    while seconds > 1e-6:
                        piece = min(seconds, 0.5)
                        f.write(f"file '{sprite}'\nduration {piece:.4f}\n")
                        seconds -= piece
                f.write(f"file '{track[-1][0]}'\n")
        jobs.append((c, name, f0, f1, teacher_txt, os.path.join(chunks_dir, f"chunk_{c:02d}.mp4")))
    with Pool(workers) as pool:
        for out in pool.imap(render_chunk, jobs):
            print(f"rendered {os.path.basename(out)}", flush=True)
    with open(os.path.join(chunks_dir, "chunks.txt"), "w") as f:
        f.writelines(f"file '{j[-1]}'\n" for j in jobs)
    out = os.path.join(B.BUILD, "lecture.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", os.path.join(chunks_dir, "chunks.txt"),
                    "-f", "concat", "-safe", "0", "-i", os.path.join(B.BUILD, "audio.txt"), "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", "-shortest", out], check=True)
    print(f"wrote {out} ({lec.duration / 60:.1f} minutes) and lecture.srt")
    return lec
