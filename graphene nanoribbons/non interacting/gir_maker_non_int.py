import os
import imageio.v2 as imageio

# ===========================================================
# USER CONTROLS
# ===========================================================
base_dir = "postproc_outputs_noninteracting/_ALL"

gif_settings = {
    "bands": {
        "folder": "bands",
        "pattern": "bands_N",
        "gif_name": "bands_vs_N.gif",
        "duration": 0.6,   # seconds per frame (↓ faster, ↑ slower)
    },
    "dos": {
        "folder": "dos",
        "pattern": "dos_N",
        "gif_name": "dos_vs_N.gif",
        "duration": 0.6,
    },
    "central": {
        "folder": "central_bands",
        "pattern": "central_N",
        "gif_name": "central_bands_vs_N.gif",
        "duration": 0.6,
    },
}
# ===========================================================


def make_gif_from_folder(folder_path, filename_pattern, gif_path, duration):
    """
    Create a GIF from PNG files in folder_path whose names start with filename_pattern.
    Files are sorted automatically.
    """
    files = sorted(
        f for f in os.listdir(folder_path)
        if f.startswith(filename_pattern) and f.endswith(".png")
    )

    if not files:
        print(f"[WARNING] No files found in {folder_path}")
        return

    images = []
    for f in files:
        img_path = os.path.join(folder_path, f)
        images.append(imageio.imread(img_path))

    # 🔁 loop=0 → infinite loop
    imageio.mimsave(
        gif_path,
        images,
        duration=duration,
        loop=0
    )

    print(f"GIF saved -> {gif_path}")



if __name__ == "__main__":
    print("\nCreating GIFs...\n")

    for key, cfg in gif_settings.items():
        folder_path = os.path.join(base_dir, cfg["folder"])
        gif_path = os.path.join(base_dir, cfg["gif_name"])

        print(f"Processing {key}...")
        make_gif_from_folder(
            folder_path=folder_path,
            filename_pattern=cfg["pattern"],
            gif_path=gif_path,
            duration=cfg["duration"],
        )

    print("\nDONE. All GIFs created.")
