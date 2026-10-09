#!/usr/bin/env python3
"""photo-dance: turn a single portrait photo into a beat-synced "dance" clip
with Persian (RTL) karaoke-style subtitles.

Pipeline (fully local, no video model needed):
  1. Cut the person out of the photo (rembg, BiRefNet portrait model).
  2. Split the cutout at the neck into head + body layers.
  3. Animate both layers on a 6/8 beat: hip sway, bounce, squash & stretch,
     shoulder shimmy and an independent head bob (Persian-dance style).
  4. Draw a party background (rotating light rays, colour cycling, confetti,
     beat flashes) and burn in RTL subtitles.
  5. Mux with audio: either your own song file (--audio) or a synthesized
     6/8 tombak-style beat.

Usage:
  python3 photo_dance.py --image friend.jpg --out dance.mp4 \
      --font Vazirmatn-Black.ttf [--audio song.mp3 --audio-start 30] \
      [--subs subs.txt] [--seconds 20]

subs.txt format: one cue per line -> "start end text" (seconds), e.g.
  3 7 بگو یار، بگو که دلم تنگ شده
"""
import argparse
import math
import os
import random
import subprocess
import sys
import tempfile
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS = 1080, 1920, 30
BPM = 120  # quarter-note pulse; 6/8 feel = 2 dotted beats per bar

DEFAULT_SUBS = [
    (0.0, 3.0, "وقتی آهنگ مورد علاقه‌ش پخش میشه..."),
    (3.0, 7.0, "بگو یار، بگو که دلم تنگ شده"),
    (7.0, 10.0, "سبیل‌ها آماده‌ی رقص!"),
    (10.0, 14.0, "بگو یار، بگو که دلم تنگ شده"),
    (14.0, 17.0, "شونه‌ها بلرزه، دست‌ها بالا"),
    (17.0, 20.0, "بگو یار، بگو که دلم تنگ شده"),
]


# ---------------------------------------------------------------- cutout
def cutout(path):
    from rembg import new_session, remove

    im = Image.open(path).convert("RGB")
    out = remove(im, session=new_session("birefnet-portrait"))
    return keep_largest_blob(out)


def keep_largest_blob(rgba):
    """Drop stray background bits (headrests, people behind) from the mask."""
    try:
        from scipy import ndimage
    except ImportError:
        return rgba
    a = np.array(rgba)
    mask = a[..., 3] > 40
    # erode first so blobs touching the person by a thin bridge separate too
    core = ndimage.binary_erosion(mask, iterations=12)
    lab, n = ndimage.label(core)
    if n <= 1:
        return rgba
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    keep = lab == (int(np.argmax(sizes)) + 1)
    keep = ndimage.binary_dilation(keep, iterations=16) & mask
    a[..., 3] = np.where(keep, a[..., 3], 0)
    return Image.fromarray(a)


def neck_y(rgba):
    """Row where the shoulders start (silhouette width jumps past the head)."""
    a = np.array(rgba)[..., 3] > 128
    h = a.shape[0]
    widths = a.sum(axis=1)
    head_w = np.median(widths[int(h * 0.2):int(h * 0.4)])
    below = np.nonzero(widths[int(h * 0.4):] > 1.6 * head_w)[0]
    return int(h * 0.4) + int(below[0]) if len(below) else int(h * 0.55)


# ---------------------------------------------------------------- motion
def beat_env(t, period, sharp=6.0):
    """1.0 right on the beat, decaying until the next one."""
    ph = (t % period) / period
    return math.exp(-sharp * ph)


def pose(t):
    q = 60.0 / BPM          # quarter note
    bar = q * 3             # 6/8 bar (two dotted-quarter beats)
    sway = math.sin(2 * math.pi * t / bar)                 # hips, 1 cycle / bar
    bounce = abs(math.sin(math.pi * t / (bar / 2)))        # 2 bounces / bar
    hit = beat_env(t, bar / 2)
    shimmy = math.sin(2 * math.pi * t * 7.0) * (0.5 + 0.5 * math.sin(t * 0.9)) ** 4
    return dict(
        angle=6.5 * sway,
        dx=55 * sway + 10 * shimmy,
        dy=-60 * bounce,
        sx=1.0 + 0.035 * hit,
        sy=1.0 - 0.045 * hit + 0.02 * bounce,
        head=38 * math.sin(2 * math.pi * t / (bar / 2) + 0.6) + 6 * shimmy,
        hit=hit,
    )


# ---------------------------------------------------------------- scene
def hsv(h, s, v):
    i = int(h * 6) % 6
    f = h * 6 - int(h * 6)
    p, q, u = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    r, g, b = [(v, u, p), (q, v, p), (p, v, u), (p, q, v), (u, p, v), (v, p, q)][i]
    return int(r * 255), int(g * 255), int(b * 255)


class Scene:
    def __init__(self, person, font_path, subs):
        self.subs = subs
        self.font = ImageFont.truetype(font_path, 74)
        self.font_title = ImageFont.truetype(font_path, 64)

        # Fit the person to ~70% of the frame height, anchored to the bottom.
        scale = (H * 0.66) / person.height
        person = person.resize((int(person.width * scale), int(person.height * scale)),
                               Image.LANCZOS)
        ny = neck_y(person)
        self.person = np.array(person)
        self.pw, self.ph = person.size
        # weight per row: 1 above the chin, easing to 0 at the shoulders, so the
        # head can slide sideways (Persian "gardan" neck slide) without seams
        ys = np.arange(self.ph, dtype=np.float32)
        top, bottom = ny - self.ph * 0.25, ny + self.ph * 0.05
        w = np.clip((bottom - ys) / (bottom - top), 0, 1)
        self.row_w = w * w * (3 - 2 * w)
        self.cols = np.arange(self.pw)

        rnd = random.Random(7)
        self.confetti = [(rnd.random() * W, rnd.random() * H, rnd.random(),
                          rnd.uniform(80, 260), rnd.uniform(6, 14)) for _ in range(140)]

        yy, xx = np.mgrid[0:H, 0:W]
        self.radial = np.clip(np.hypot(xx - W / 2, yy - H * 0.42) / (H * 0.75), 0, 1)
        self.theta = np.arctan2(yy - H * 0.42, xx - W / 2)

    def background(self, t, hit):
        hue = (t * 0.05) % 1
        c1 = np.array(hsv(hue, 0.85, 1.0), np.float32)
        c2 = np.array(hsv((hue + 0.55) % 1, 0.9, 0.25), np.float32)
        r = self.radial[..., None]
        img = c1 * (1 - r) + c2 * r
        rays = (np.sin(self.theta * 12 + t * 1.6) > 0.55)[..., None]
        img = img + rays * (40 + 60 * hit)
        img = np.clip(img * (0.85 + 0.25 * hit), 0, 255).astype(np.uint8)
        bg = Image.fromarray(img, "RGB").convert("RGBA")
        d = ImageDraw.Draw(bg)
        for x, y, h, sp, sz in self.confetti:
            yy = (y + sp * t) % (H + 40) - 20
            xx = x + 25 * math.sin(t * 2 + h * 10)
            d.rectangle([xx, yy, xx + sz, yy + sz * 0.5], fill=hsv(h, 0.8, 1.0))
        return bg

    def person_frame(self, t, p):
        canvas = Image.new("RGBA", (self.pw * 2, self.ph * 2), (0, 0, 0, 0))
        ox, oy = self.pw // 2, self.ph // 2
        shift = (p["head"] * self.row_w).astype(np.int32)
        src_x = np.clip(self.cols[None, :] - shift[:, None], 0, self.pw - 1)
        warped = np.take_along_axis(self.person, src_x[..., None], axis=1)
        canvas.alpha_composite(Image.fromarray(warped, "RGBA"), (ox, oy))
        # whole-figure sway around the hips (bottom centre)
        pivot = (ox + self.pw / 2, oy + self.ph)
        canvas = canvas.rotate(p["angle"], resample=Image.BICUBIC, center=pivot)
        nw, nh = int(canvas.width * p["sx"]), int(canvas.height * p["sy"])
        canvas = canvas.resize((nw, nh), Image.BILINEAR)
        x = int(W / 2 - nw / 2 + p["dx"])
        y = int(H + 40 - (oy + self.ph) * p["sy"] + p["dy"])
        return canvas, (x, y)

    def text(self, img, s, y, font, fill, stroke):
        d = ImageDraw.Draw(img)
        bb = d.textbbox((0, 0), s, font=font, direction="rtl", language="fa")
        while bb[2] - bb[0] > W - 80 and font.size > 30:
            font = font.font_variant(size=font.size - 4)
            bb = d.textbbox((0, 0), s, font=font, direction="rtl", language="fa")
        tw = bb[2] - bb[0]
        d.text(((W - tw) / 2 - bb[0], y), s, font=font, fill=fill,
               stroke_width=stroke, stroke_fill=(0, 0, 0), direction="rtl", language="fa")
        return bb

    def render(self, t):
        p = pose(t)
        frame = self.background(t, p["hit"])
        fig, pos = self.person_frame(t, p)
        # glow shadow behind the dancer
        glow = Image.new("RGBA", fig.size, hsv((t * 0.05 + 0.5) % 1, 1, 1) + (0,))
        glow.putalpha(fig.getchannel("A").filter(ImageFilter.GaussianBlur(18)))
        frame.alpha_composite(glow, (pos[0], pos[1]))
        frame.alpha_composite(fig, pos)

        self.text(frame, "بگو یار", 120, self.font_title, (255, 255, 255), 6)

        for start, end, line in self.subs:
            if start <= t < end:
                pop = min(1.0, (t - start) / 0.25)
                y = H - 330 + int(40 * (1 - pop))
                band = Image.new("RGBA", (W, 190), (0, 0, 0, int(150 * pop)))
                frame.alpha_composite(band, (0, H - 360))
                col = (255, 230, 60) if "بگو یار" in line else (255, 255, 255)
                self.text(frame, line, y, self.font, col, 5)
                break
        return frame.convert("RGB")


# ---------------------------------------------------------------- audio
def synth_beat(seconds, sr=44100):
    """6/8 tombak-style groove (tom ... bak bak) + bass + small Homayoun riff."""
    n = int(seconds * sr)
    out = np.zeros(n, np.float32)
    rng = np.random.default_rng(1)
    e = 60.0 / BPM / 2  # eighth note

    def add(at, sig):
        i = int(at * sr)
        j = min(n, i + len(sig))
        if i < n:
            out[i:j] += sig[: j - i]

    def tone(freq, dur, decay, amp, sweep=0.0):
        tt = np.arange(int(dur * sr)) / sr
        f = freq * (1 + sweep * np.exp(-tt * 40))
        ph = 2 * np.pi * np.cumsum(f) / sr
        return (amp * np.sin(ph) * np.exp(-tt * decay)).astype(np.float32)

    def noise(dur, decay, amp):
        tt = np.arange(int(dur * sr)) / sr
        s = rng.standard_normal(len(tt)) * np.exp(-tt * decay)
        s = np.diff(s, prepend=0)  # crude high-pass -> "bak" snap
        return (amp * s).astype(np.float32)

    # Homayoun-ish scale on D: D Eb F# G A Bb C
    scale = [293.66, 311.13, 369.99, 392.0, 440.0, 466.16, 523.25, 587.33]
    riff = [4, 3, 2, 3, 4, None, 5, 4, 3, 2, 1, 0]
    bar = 6 * e
    t = 0.0
    k = 0
    while t < seconds:
        # tombak: DUM . bak bak DUM bak
        add(t, tone(70, 0.5, 9, 0.9, sweep=1.5))
        add(t + 2 * e, noise(0.12, 45, 0.35))
        add(t + 3 * e, tone(70, 0.45, 9, 0.7, sweep=1.5))
        add(t + 4 * e, noise(0.12, 45, 0.30))
        add(t + 5 * e, noise(0.12, 45, 0.40))
        add(t + 3 * e, noise(0.08, 60, 0.25))  # clap
        # bass
        add(t, tone(73.42, bar * 0.48, 3, 0.35))
        add(t + 3 * e, tone(98.0 if k % 2 else 110.0, bar * 0.48, 3, 0.3))
        # lead riff (plucked, with 2nd harmonic)
        for i in range(6):
            note = riff[(k * 6 + i) % len(riff)]
            if note is not None:
                f = scale[note]
                add(t + i * e, tone(f, e * 1.4, 7, 0.18) + tone(f * 2, e * 1.4, 10, 0.06))
        t += bar
        k += 1
    out /= max(1e-6, np.abs(out).max()) / 0.85
    fade = int(sr * 1.0)
    out[-fade:] *= np.linspace(1, 0, fade)
    return (out * 32767).astype(np.int16), sr


def write_wav(path, data, sr):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(data.tobytes())


# ---------------------------------------------------------------- main
def load_subs(path):
    subs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                a, b, txt = line.split(maxsplit=2)
                subs.append((float(a), float(b), txt))
    return subs


def write_srt(path, subs):
    def ts(x):
        ms = int(round(x * 1000))
        return f"{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02},{ms % 1000:03}"
    with open(path, "w", encoding="utf-8") as f:
        for i, (a, b, txt) in enumerate(subs, 1):
            f.write(f"{i}\n{ts(a)} --> {ts(b)}\n{txt}\n\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="dance.mp4")
    ap.add_argument("--font", required=True, help="Persian TTF, e.g. Vazirmatn-Black.ttf")
    ap.add_argument("--audio", help="your own song file; omit to synthesize a 6/8 beat")
    ap.add_argument("--audio-start", type=float, default=0.0,
                    help="seconds into --audio where the clip should start")
    ap.add_argument("--subs", help="cue file: 'start end text' per line")
    ap.add_argument("--seconds", type=float, default=20.0)
    args = ap.parse_args()

    subs = load_subs(args.subs) if args.subs else DEFAULT_SUBS
    scene = Scene(cutout(args.image), args.font, subs)
    tmp = tempfile.mkdtemp(prefix="photo-dance-")
    silent = os.path.join(tmp, "video.mp4")

    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
         "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", silent],
        stdin=subprocess.PIPE)
    total = int(args.seconds * FPS)
    for i in range(total):
        enc.stdin.write(scene.render(i / FPS).tobytes())
        if i % FPS == 0:
            print(f"\rrendering {i / FPS:4.0f}s / {args.seconds:.0f}s", end="", file=sys.stderr)
    enc.stdin.close()
    enc.wait()
    print(file=sys.stderr)

    if args.audio:
        audio_in = ["-ss", str(args.audio_start), "-i", args.audio]
    else:
        wav = os.path.join(tmp, "beat.wav")
        write_wav(wav, *synth_beat(args.seconds))
        audio_in = ["-i", wav]
    subprocess.check_call(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", silent, *audio_in,
         "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
         "-af", f"afade=t=out:st={args.seconds - 1}:d=1", "-t", str(args.seconds),
         "-movflags", "+faststart", args.out])
    write_srt(os.path.splitext(args.out)[0] + ".srt", subs)
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
