# Narration clips

`manifest.jsonl` lists every narration clip of the lecture, in order, one JSON object per line:
`index`, `key`, `chapter`, `kind` (narration, exercise or predict) and `text` (exactly what to say).
Regenerate it with `python lecture/build.py manifest` after editing `lecture/lecture.md`.

To use audio made with any TTS model, put one file per clip in `lecture/tts/<voice-name>/<key>.mp3`
(or .wav/.flac/.ogg), then build with that folder:

    python lecture/build.py audio --tts=files --voice=<voice-name>
    python lecture/build.py video --tts=files --voice=<voice-name>

The key is a hash of the clip's text, so edited paragraphs get new keys and unchanged ones are reused.

## ElevenLabs, from GitHub Actions

`.github/workflows/tts.yml` generates the clips with the repository secret `ELEVENLABS_API_KEY` and uploads them as
a workflow artifact called `narration` (nothing gets committed). Run it from the Actions tab, or with the API:

- `mode=list`: what the key can see (voices; the quota and models too, if the key has those permissions)
- `mode=generate`, `voices=<voice id>`, `indices=all` (or e.g. `0,93` for an audition)

Then unzip the artifact into `lecture/tts/` (it holds one folder per voice, `elevenlabs-<model>-<voice>/`) and build
with `--tts=files --voice=elevenlabs-<model>-<voice>`.

## Long recordings (browser TTS, e.g. Google AI Studio)

`script/` has the narration as plain text, one file per part of a chapter (at most ~3,000 characters, 3-4
minutes of speech), with a blank line between paragraphs. Read each file out in one go, save the recording
with the same name (`chapter_04_the-data-mixture_part2.wav`), put them all in one folder, and run

    python lecture/build.py import --parts <folder> --voice <voice-name>

It finds where every paragraph starts with Whisper, cuts at the quietest moment in between, and writes
`lecture/tts/<voice-name>/<key>.mp3` for every clip, ready for `--tts=files --voice=<voice-name>`.
