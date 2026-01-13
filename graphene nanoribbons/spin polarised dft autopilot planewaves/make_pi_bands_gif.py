# make_magn_profile_gifs.py
#
# Makes TWO GIFs:
#   1) magn_profile_clean.gif      (no brackets)
#   2) magn_profile_annotated.gif  (with brackets)
#
# Uses Pillow to set duration in ms (more reliable than imageio in many viewers)
# and repeats frames to force slowdown if a viewer ignores timing metadata.

import os
import glob
from PIL import Image

# -----------------------------
# USER SETTINGS
# -----------------------------
SUBFOLDER = "central band selection, fitting and band magnetisation graphs"
FRAME_DIR = "magn_profile"

CLEAN_NAME = "magn_profile_clean.png"
ANN_NAME   = "magn_profile_annotated.png"

OUT_SUBDIR = "_ALL/magn_profile"
OUT_GIF_CLEAN = "magn_profile_clean.gif"
OUT_GIF_ANN   = "magn_profile_annotated.gif"

# Desired seconds per slide:
SECONDS_PER_SLIDE = 0.9

# If your viewer still plays too fast, increase this (2, 3, 4...)
# This repeats each slide N times, forcing longer playback.
REPEAT_EACH_FRAME = 2

# Loop forever
LOOP = 0
# -----------------------------


def resolve_sweep_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(here, "zgnr_sweep_results")
    if os.path.isdir(cand):
        return cand

    cwd = os.getcwd()
    cand2 = os.path.join(cwd, "zgnr_sweep_results")
    if os.path.isdir(cand2):
        return cand2

    if any(os.path.isdir(p) for p in glob.glob(os.path.join(cwd, "Ny*"))):
        return cwd

    return None


def ny_key(path):
    base = os.path.basename(path).lower()
    if base.startswith("ny"):
        try:
            return int(base[2:])
        except ValueError:
            return 10**9
    return 10**9


def collect_frames(sweep_dir, filename):
    ny_dirs = [p for p in glob.glob(os.path.join(sweep_dir, "Ny*")) if os.path.isdir(p)]
    ny_dirs.sort(key=ny_key)

    frames = []
    for ny in ny_dirs:
        p = os.path.join(ny, SUBFOLDER, FRAME_DIR, filename)
        if os.path.isfile(p):
            frames.append(p)
    return frames


def write_gif_pillow(frame_paths, out_path, seconds_per_slide, repeat_each=1):
    if not frame_paths:
        print(f"[ERROR] No frames found for: {out_path}")
        return

    duration_ms = int(round(seconds_per_slide * 1000))

    # Load images as PIL and convert to palette mode (good for GIF)
    imgs = []
    for p in frame_paths:
        im = Image.open(p).convert("RGBA")
        # Convert to paletted for GIF (keeps file small & compatible)
        im = im.convert("P", palette=Image.Palette.ADAPTIVE)
        for _ in range(max(1, int(repeat_each))):
            imgs.append(im)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Save GIF with duration in milliseconds
    imgs[0].save(
        out_path,
        save_all=True,
        append_images=imgs[1:],
        duration=duration_ms,
        loop=LOOP,
        optimize=False,
        disposal=2
    )

    print(f"[OK] Wrote GIF: {out_path}")
    print(f"     base frames: {len(frame_paths)} | repeated: x{repeat_each} -> {len(imgs)} frames")
    print(f"     duration/frame: {duration_ms} ms (requested {seconds_per_slide:.2f} s)")


def main():
    sweep_dir = resolve_sweep_dir()
    if sweep_dir is None:
        print("[ERROR] Could not locate zgnr_sweep_results.")
        return

    out_dir = os.path.join(sweep_dir, OUT_SUBDIR)
    out_clean = os.path.join(out_dir, OUT_GIF_CLEAN)
    out_ann = os.path.join(out_dir, OUT_GIF_ANN)

    clean_frames = collect_frames(sweep_dir, CLEAN_NAME)
    ann_frames   = collect_frames(sweep_dir, ANN_NAME)

    print(f"[INFO] Clean frames found: {len(clean_frames)}")
    print(f"[INFO] Annotated frames found: {len(ann_frames)}")

    if clean_frames:
        write_gif_pillow(clean_frames, out_clean, SECONDS_PER_SLIDE, repeat_each=REPEAT_EACH_FRAME)
    else:
        print("[ERROR] No CLEAN frames found (magn_profile_clean.png).")

    if ann_frames:
        write_gif_pillow(ann_frames, out_ann, SECONDS_PER_SLIDE, repeat_each=REPEAT_EACH_FRAME)
    else:
        print("[ERROR] No ANNOTATED frames found (magn_profile_annotated.png).")

    print("DONE.")


if __name__ == "__main__":
    main()
