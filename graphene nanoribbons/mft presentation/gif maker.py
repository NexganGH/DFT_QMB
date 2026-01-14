import os
import glob
from PIL import Image

BASE_DIR = "postproc_outputs/_ALL"
FPS = 6
LOOP = 0  # 0 = forever


def sorted_pngs(folder):
    files = glob.glob(os.path.join(folder, "*.png"))
    files.sort()  # relies on N being in filename with leading zeros
    return files


def pad_to_size(img, target_size, bg=(255, 255, 255)):
    W, H = target_size
    img = img.convert("RGB")
    canvas = Image.new("RGB", (W, H), bg)
    x = (W - img.size[0]) // 2
    y = (H - img.size[1]) // 2
    canvas.paste(img, (x, y))
    return canvas


def make_gif(in_folder, out_path, fps=6, loop=0):
    pngs = sorted_pngs(in_folder)
    print(f"\nFolder: {in_folder}")
    print(f"Found {len(pngs)} PNG frames")

    if len(pngs) < 2:
        print("[WARN] Need at least 2 frames for an evolving GIF.")
        if len(pngs) == 1:
            print("       Only frame:", pngs[0])
        return

    frames = [Image.open(p).convert("RGB") for p in pngs]
    max_w = max(im.size[0] for im in frames)
    max_h = max(im.size[1] for im in frames)
    target = (max_w, max_h)

    frames = [pad_to_size(im, target) for im in frames]

    duration_ms = int(round(1000 / fps))
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=loop,
        optimize=False,
    )

    print(f"[OK] Saved GIF: {out_path}")
    print(f"     Frame size: {target[0]} x {target[1]}")
    print(f"     Duration per frame: {duration_ms} ms")


if __name__ == "__main__":
    make_gif(os.path.join(BASE_DIR, "bands"), os.path.join(BASE_DIR, "bands.gif"), fps=FPS, loop=LOOP)
    make_gif(os.path.join(BASE_DIR, "dos"), os.path.join(BASE_DIR, "dos.gif"), fps=FPS, loop=LOOP)
    make_gif(os.path.join(BASE_DIR, "magnetization"), os.path.join(BASE_DIR, "magnetization.gif"), fps=FPS, loop=LOOP)

    print("\nDONE.")
