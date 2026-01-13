import matplotlib.pyplot as plt
from common.mpl_style import set_mpl_style

set_mpl_style()

import matplotlib.pyplot as plt

# -----------------------------
# Geometry (units arbitrary)
# -----------------------------
d = 1.0

z_bottom = 0.0
z_top    = d
z_gate   = 2.0 * d   # z0 = 2d

x_min, x_max = -5, 5

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 3))

# Layers
ax.plot([x_min, x_max], [z_bottom, z_bottom],
        linewidth=4, color="#2A9D8F", label="Bottom layer")

ax.plot([x_min, x_max], [z_top, z_top],
        linewidth=4, color="#E9C46A", label="Top layer")

ax.plot([x_min, x_max], [z_gate, z_gate],
        linewidth=3, linestyle="--", color="#555555", label="Gate")

# -----------------------------
# Distance annotations
# -----------------------------
ax.annotate("", xy=(0, z_top), xytext=(0, z_bottom),
            arrowprops=dict(arrowstyle="<->", linewidth=1.3))
ax.text(0.1, 0.5 * d, r"$d$", va="center", fontsize=13)

ax.annotate("", xy=(0, z_gate), xytext=(0, z_top),
            arrowprops=dict(arrowstyle="<->", linewidth=1.3))
ax.text(0.1, 1.5 * d, r"$d$", va="center", fontsize=13)

ax.annotate("", xy=(x_max - 0.5, z_gate), xytext=(x_max - 0.5, z_bottom),
            arrowprops=dict(arrowstyle="<->", linewidth=1.3))
ax.text(x_max - 0.3, d, r"$z_0 = 2d$", va="center", fontsize=13)

# -----------------------------
# Axis styling
# -----------------------------
# Keep only the z-axis (left spine)
ax.spines["left"].set_visible(True)
ax.spines["left"].set_position(("data", x_min))

# Remove other spines
for spine in ["top", "right", "bottom"]:
    ax.spines[spine].set_visible(False)

# z-axis arrow
ax.annotate("", xy=(x_min, z_gate + 0.5 * d),
            xytext=(x_min, z_bottom - 0.3 * d),
            arrowprops=dict(arrowstyle="->", linewidth=1.5))
ax.text(x_min - 0.4, z_gate + 0.5 * d, r"$z$", fontsize=14)

# Ticks
ax.set_xticks([])
ax.set_yticks([])

# Limits
ax.set_xlim(x_min - 0.8, x_max + 1.8)
ax.set_ylim(z_bottom - 0.4 * d, z_gate + 0.7 * d)

# -----------------------------
# Legend (outside, no overlap)
# -----------------------------
ax.legend(loc="center left",
          bbox_to_anchor=(1.02, 0.5),
          frameon=False,
          fontsize=13)

plt.tight_layout()
plt.show()
