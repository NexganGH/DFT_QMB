import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

def draw_zigzag_strands_trimmed(
    Ny=4,
    Nx=6,
    dx=1.6,
    dy=0.55,
    row_gap=1.35,
    sq_size=0.22,
    circ_r=0.16,
    bond_lw=1.3,
    square_color="#2E8B57",
    circle_color="#F2B179",
    bond_color="#3a3a3a",
    savepath=None
):
    fig, ax = plt.subplots(figsize=(10, 4.8))

    for r in range(Ny):
        shift = (r % 2) * (dx / 2.0)

        y_sq = (Ny - 1 - r) * row_gap
        y_c  = y_sq - dy

        # Full set of circle positions (j=0..Nx)
        circle_pos = [(j, j * dx + shift) for j in range(Nx + 1)]
        # Squares between j and j+1 (j=0..Nx-1)
        square_pos = [(j, (j + 0.5) * dx + shift) for j in range(Nx)]

        # Remove the "purple-cancelled" end circle:
        # even rows -> remove leftmost circle (j=0)
        # odd rows  -> remove rightmost circle (j=Nx)
        circles_present = {j for j, _ in circle_pos}
        if r % 2 == 0:
            circles_present.remove(0)
        else:
            circles_present.remove(Nx)

        # Draw bonds: only if the endpoint circle exists
        for j, xs in square_pos:
            if j in circles_present:
                ax.plot([xs, j * dx + shift], [y_sq, y_c],
                        color=bond_color, lw=bond_lw, zorder=1)
            if (j + 1) in circles_present:
                ax.plot([xs, (j + 1) * dx + shift], [y_sq, y_c],
                        color=bond_color, lw=bond_lw, zorder=1)

        # Draw circles (only the ones kept)
        for j, x in circle_pos:
            if j not in circles_present:
                continue
            ax.add_patch(Circle((x, y_c), radius=circ_r,
                                facecolor=circle_color, edgecolor=bond_color,
                                linewidth=1.0, zorder=3))

        # Draw squares (all of them)
        for _, x in square_pos:
            ax.add_patch(Rectangle((x - sq_size, y_sq - sq_size),
                                   2 * sq_size, 2 * sq_size,
                                   facecolor=square_color, edgecolor=bond_color,
                                   linewidth=1.0, zorder=4))

    ax.set_aspect("equal")
    ax.axis("off")

    xmin = -0.5 * dx
    xmax = (Nx + 0.5) * dx + dx/2
    ymin = -dy - 0.6
    ymax = (Ny - 1) * row_gap + 0.8
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    draw_zigzag_strands_trimmed(Ny=6, Nx=6, savepath="zigzag_trimmed.png")
