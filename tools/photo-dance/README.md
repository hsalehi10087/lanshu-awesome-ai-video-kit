# 🕺 photo-dance — one photo → beat-synced dance clip with Persian subtitles

Turns a single portrait/selfie into a 1080×1920 (Reels/Shorts) clip where the
person dances on a 6/8 Persian beat, with RTL Persian karaoke-style subtitles
burned in plus a separate `.srt`. Runs fully locally; no video model needed.

What it does:

- Cuts the person out with `rembg` (BiRefNet portrait model)
- Hip sway + bounce + squash & stretch on the beat
- Persian **neck slide** (گردن) with a seamless row warp, so there's no visible cut at the neck
- Party background with rotating rays, colour cycling, confetti and beat flashes
- RTL subtitles via Pillow + libraqm, shrunk automatically to fit the frame width

## Install

```bash
pip install "rembg[cpu]" scipy pillow numpy
# ffmpeg must be on PATH; a Persian font, e.g. Vazirmatn:
# https://github.com/rastikerdar/vazirmatn/releases
```

## Run

```bash
# with a synthesized 6/8 tombak groove (no copyrighted audio)
python3 photo_dance.py --image friend.jpg --font Vazirmatn-Black.ttf --out dance.mp4

# with your own copy of the song, starting at 0:42 into the track
python3 photo_dance.py --image friend.jpg --font Vazirmatn-Black.ttf \
    --audio "begoo-yar.mp3" --audio-start 42 --out dance.mp4

# custom subtitles: one "start end text" cue per line
python3 photo_dance.py ... --subs subs.txt --seconds 25
```

`subs.txt` example:

```
0 3 وقتی آهنگ مورد علاقه‌ش پخش میشه...
3 7 بگو یار، بگو که دلم تنگ شده
```

To swap the audio on an already rendered clip:

```bash
ffmpeg -i dance.mp4 -ss 42 -i song.mp3 -map 0:v -map 1:a -c:v copy -shortest out.mp4
```

## Want real full-body dancing?

This tool only animates the existing photo (2.5D). For real limb and body
motion, feed the same photo to an image-to-video model (Kling 3.0 I2V,
Wan 2.7, Hailuo 02) with a prompt like:

> The man in the photo stands up and dances a joyful Persian/Iranian dance
> to a 6/8 rhythm: graceful wrist rotations, snapping fingers above his head,
> shoulder shimmies, playful eyebrow raises and a neck slide, big smile under
> his thick mustache. Colourful party lights, confetti, warm festive mood.
> Medium shot, steady camera, slight push-in. Keep his face, hair, mustache
> and grey graphic T-shirt identical to the reference.

Then run the result through ffmpeg to add the song and the `.srt` subtitles.

> ⚠️ Only use photos of people who are fine with it, and don't publish
> copyrighted music without the rights to it.
