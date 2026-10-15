# Narration clips

`manifest.jsonl` lists every narration clip of the lecture, in order, one JSON object per line:
`index`, `key`, `chapter`, `kind` (narration or exercise) and `text` (exactly what to say).
Regenerate it with `python lecture/build.py manifest` after editing `lecture/lecture.md`.

To use audio made with any TTS model, put one file per clip in `lecture/tts/<voice-name>/<key>.mp3`
(or .wav/.flac/.ogg), then build with that folder:

    python lecture/build.py audio --tts=files --voice=<voice-name>
    python lecture/build.py video --tts=files --voice=<voice-name>

The key is a hash of the clip's text, so edited paragraphs get new keys and unchanged ones are reused.
