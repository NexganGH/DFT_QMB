# gif_for_grid_magn.py
#
# Robust looping GIF maker:
# - pads frames so all have same pixel size
# - repeats frames to guarantee "seconds per slide" (works even if viewers clamp long delays)

import os
import re
import numpy as np
from PIL import Image

# ===========================================================
# USER SETTINGS
# ===========================================================
BASE_DIR = "postproc_outputs/_ALL/lattice_mag_fixed"   # folder containing PNG frames
GIF_NAME = "lattice_mag_fixed.gif"

SECONDS_PER_SLIDE = 1   # <-- what you want (e.g. 2.0, 5.0, 10.0)
BASE_STEP_SEC = 0.10       # internal per-frame delay (keep small & stable)
LOOP = 0                   # 0 = infinite loop

PINGPONG = False           # True -> forward then backward (nice for talks)
# ===========================================================


def natural_sort_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def pad_to_shape(img_arr, target_h, target_w, bg=255):
    """
    Center-pad an image array to (target_h, target_w).
    bg=255 -> white padding.
    """
    if img_arr.ndim == 2:
        h, w = img_arr.shape
        c = None
    else:
        h, w, c = img_arr.shape

    top = (target_h - h) // 2
    left = (target_w - w) // 2

    if c is None:
        out = np.full((target_h, target_w), bg, dtype=img_arr.dtype)
        out[top:top+h, left:left+w] = img_arr
    else:
        out = np.full((target_h, target_w, c), bg, dtype=img_arr.dtype)
        out[top:top+h, left:left+w, :] = img_arr

    return out


def load_and_pad_frames(folder):
    files = sorted(
        [f for f in os.listdir(folder) if f.lower().endswith(".png")],
        key=natural_sort_key
    )
    if not files:
        raise RuntimeError(f"No PNG files found in: {folder}")

    # read all as arrays, find max dims
    arrays = []
    max_h, max_w = 0, 0

    for f in files:
        p = os.path.join(folder, f)
        im = Image.open(p).convert("RGB")
        arr = np.array(im)
        arrays.append(arr)
        h, w = arr.shape[:2]
        max_h = max(max_h, h)
        max_w = max(max_w, w)

    # pad all to max dims
    padded = [pad_to_shape(a, max_h, max_w, bg=255) for a in arrays]

    # convert back to PIL Images
    frames = [Image.fromarray(a) for a in padded]
    return files, frames, (max_w, max_h)


def make_gif(folder, gif_path, seconds_per_slide, base_step_sec, loop, pingpong):
    files, frames, (W, H) = load_and_pad_frames(folder)

    if pingpong and len(frames) > 1:
        frames = frames + frames[-2:0:-1]   # forward then backward, no duplicate endpoints

    # Repeat each frame R times to guarantee long hold times
    # Effective per-slide time ≈ R * base_step_sec
    if base_step_sec <= 0:
        raise ValueError("BASE_STEP_SEC must be > 0")

    repeat = max(1, int(round(seconds_per_slide / base_step_sec)))
    duration_ms = int(round(base_step_sec * 1000))

    expanded = []
    for fr in frames:
        expanded.extend([fr] * repeat)

    # Save GIF (Pillow)
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    expanded[0].save(
        gif_path,
        save_all=True,
        append_images=expanded[1:],
        duration=duration_ms,  # milliseconds per internal frame
        loop=loop,
        optimize=False,
        disposal=2
    )

    n_unique = len(frames)
    n_total = len(expanded)
    eff = repeat * base_step_sec

    print(f"GIF saved -> {gif_path}")
    print(f"Folder: {folder}")
    print(f"Frame size: {W}x{H}")
    print(f"Unique frames: {n_unique} (pingpong={pingpong})")
    print(f"Repeat per unique frame: {repeat} (base step = {base_step_sec:.3f}s)")
    print(f"Effective seconds per slide ≈ {eff:.3f}s")
    print(f"Total GIF frames written: {n_total}")
    print("Loop:", "infinite" if loop == 0 else loop)


if __name__ == "__main__":
    if not os.path.isdir(BASE_DIR):
        raise FileNotFoundError(f"Folder not found: {BASE_DIR}")

    out_dir = os.path.dirname(BASE_DIR)          # .../_ALL
    gif_path = os.path.join(out_dir, GIF_NAME)   # .../_ALL/lattice_mag_fixed.gif

    make_gif(
        folder=BASE_DIR,
        gif_path=gif_path,
        seconds_per_slide=SECONDS_PER_SLIDE,
        base_step_sec=BASE_STEP_SEC,
        loop=LOOP,
        pingpong=PINGPONG
    )
