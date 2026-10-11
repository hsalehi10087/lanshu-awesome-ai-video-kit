#!/usr/bin/env python3
"""finish_clip: add a song and Persian (RTL) subtitles to a clip made elsewhere,
e.g. an image-to-video result from Kling / Wan / Hailuo.

  python3 finish_clip.py --clip kling.mp4 --font Vazirmatn-Black.ttf \
      --subs subs.txt --out final.mp4 [--audio song.mp3 --audio-start 42]

The clip is scaled and cropped to 1080x1920. Its own audio is dropped unless
no --audio is given. subs.txt format: "start end text" per line (seconds).
"""
import argparse
import os
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw, ImageFont

W, H = 1080, 1920


def load_subs(path):
    subs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                a, b, txt = line.split(maxsplit=2)
                subs.append((float(a), float(b), txt))
    return subs


def fit_text(draw, s, font):
    """Shrink the font until the line fits the frame width."""
    while True:
        bb = draw.textbbox((0, 0), s, font=font, direction="rtl", language="fa")
        if bb[2] - bb[0] <= W - 80 or font.size <= 30:
            return font, bb
        font = font.font_variant(size=font.size - 4)


def render_cue(path, text, font_path, is_hook):
    """Transparent full-frame PNG with the subtitle band at the lower third."""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    font, bb = fit_text(d, text, ImageFont.truetype(font_path, 74))
    tw = bb[2] - bb[0]
    y = H - 330
    d.rectangle([0, H - 360, W, H - 170], fill=(0, 0, 0, 150))
    fill = (255, 230, 60) if is_hook else (255, 255, 255)
    d.text(((W - tw) / 2 - bb[0], y), text, font=font, fill=fill,
           stroke_width=5, stroke_fill=(0, 0, 0), direction="rtl", language="fa")
    img.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--font", required=True)
    ap.add_argument("--subs", required=True)
    ap.add_argument("--out", default="final.mp4")
    ap.add_argument("--audio")
    ap.add_argument("--audio-start", type=float, default=0.0)
    args = ap.parse_args()

    subs = load_subs(args.subs)
    tmp = tempfile.mkdtemp(prefix="finish-clip-")
    cues = []
    for i, (a, b, txt) in enumerate(subs):
        p = os.path.join(tmp, f"cue{i}.png")
        render_cue(p, txt, args.font, "بگو یار" in txt)
        cues.append((a, b, p))

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", args.clip]
    for _, _, p in cues:
        cmd += ["-i", p]
    audio_idx = None
    if args.audio:
        cmd += ["-ss", str(args.audio_start), "-i", args.audio]
        audio_idx = len(cues) + 1

    # scale + crop to 9:16, then stack the subtitle overlays with enable windows
    chain = (f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,"
             f"crop={W}:{H},setsar=1[v0]")
    prev = "v0"
    for i, (a, b, _) in enumerate(cues, 1):
        out = f"v{i}"
        chain += (f";[{prev}][{i}:v]overlay=0:0:"
                  f"enable='between(t,{a},{b})'[{out}]")
        prev = out
    fc = chain
    maps = ["-map", f"[{prev}]"]
    if audio_idx is not None:
        maps += ["-map", f"{audio_idx}:a", "-c:a", "aac", "-b:a", "192k"]
    else:
        maps += ["-an"]
    cmd += ["-filter_complex", fc, *maps, "-c:v", "libx264", "-preset", "medium",
            "-crf", "20", "-pix_fmt", "yuv420p", "-shortest", "-movflags", "+faststart",
            args.out]
    subprocess.check_call(cmd)

    srt = os.path.splitext(args.out)[0] + ".srt"
    with open(srt, "w", encoding="utf-8") as f:
        for i, (a, b, txt) in enumerate(subs, 1):
            f.write(f"{i}\n{ts(a)} --> {ts(b)}\n{txt}\n\n")
    print(f"done -> {args.out}", file=sys.stderr)


def ts(x):
    ms = int(round(x * 1000))
    return f"{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02},{ms % 1000:03}"


if __name__ == "__main__":
    main()
