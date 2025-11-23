import numpy as np
import pyvista as pv
from gpaw import GPAW

# ----------------------------
# Load DFT and extract density
# ----------------------------
calc = GPAW("graphene_lda_scf.gpw")
rho = calc.get_all_electron_density(gridrefinement=1)
nx, ny, nz = rho.shape
cell = calc.atoms.cell[:]

# ----------------------------
# Build correct crystal grid
# ----------------------------
a1, a2, a3 = cell

i = np.linspace(0, 1, nx)
j = np.linspace(0, 1, ny)
k = np.linspace(0, 1, nz)

ii, jj, kk = np.meshgrid(i, j, k, indexing="ij")

xx = ii * a1[0] + jj * a2[0] + kk * a3[0]
yy = ii * a1[1] + jj * a2[1] + kk * a3[1]
zz = ii * a1[2] + jj * a2[2] + kk * a3[2]

grid = pv.StructuredGrid(xx, yy, zz)
grid["rho"] = rho.flatten(order="F")

# ----------------------------
# PyVista Plotter
# ----------------------------
plotter = pv.Plotter()

# Draw atoms
for pos in calc.atoms.positions:
    sphere = pv.Sphere(radius=0.2, center=pos)
    plotter.add_mesh(sphere, color="white")

# This will store the ACTOR returned by add_mesh()
iso_actor = None

# ----------------------------
# Callback for the slider
# ----------------------------
def update_isovalue(isovalue):
    global iso_actor

    # If an actor already exists → remove it safely
    if iso_actor is not None:
        plotter.remove_actor(iso_actor)

    # Compute new isosurface
    iso_mesh = grid.contour([isovalue])

    # Add surface and store the ACTOR
    actor = plotter.add_mesh(iso_mesh, color="cyan", opacity=0.5)
    iso_actor = actor    # save for next update

    plotter.render()


# ----------------------------
# Add slider widget
# ----------------------------
vmin = float(np.percentile(rho, 60))
vmax = float(np.percentile(rho, 99))
initial = float(np.percentile(rho, 90))

plotter.add_slider_widget(
    update_isovalue,
    rng=[vmin, vmax],
    value=initial,
    title="Isosurface level (electron density)"
)

# First render
update_isovalue(initial)

plotter.show()
