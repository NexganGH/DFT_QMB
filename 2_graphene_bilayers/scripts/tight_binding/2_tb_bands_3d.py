import numpy as np
import plotly.graph_objects as go

# Tight-binding parameters (eV)
gamma0 = 3.1
gamma1 = 0.39
gamma3 = 0.315
gamma4 = 0.044

# Lattice constants
a = 1.42
a0 = np.sqrt(3) * a

# Nearest-neighbour vectors
delta = np.array([
    [0,  a/np.sqrt(3)],
    [ a/2, -a/(2*np.sqrt(3))],
    [-a/2, -a/(2*np.sqrt(3))]
])

def f_k(kx, ky):
    return np.sum(np.exp(1j*(delta[:,0]*kx + delta[:,1]*ky)))

def make_H(kx, ky):
    f = f_k(kx, ky)
    H = np.zeros((4,4), dtype=complex)

    H[0,0] = 0
    H[1,1] = 0
    H[2,2] = 0
    H[3,3] = 0

    H[0,1] = -gamma0 * f
    H[1,0] = -gamma0 * np.conj(f)
    H[2,3] = -gamma0 * f
    H[3,2] = -gamma0 * np.conj(f)

    H[1,2] = gamma1
    H[2,1] = gamma1

    H[0,3] = -gamma3 * np.conj(f)
    H[3,0] = -gamma3 * f

    H[0,2] = gamma4 * f
    H[2,0] = gamma4 * np.conj(f)
    H[1,3] = gamma4 * f
    H[3,1] = gamma4 * np.conj(f)

    return H

# Grid
Nk = 60
kx_vals = np.linspace(-2*np.pi/a0, 2*np.pi/a0, Nk)
ky_vals = np.linspace(-2*np.pi/a0, 2*np.pi/a0, Nk)
KX, KY = np.meshgrid(kx_vals, ky_vals)

bands = np.zeros((Nk, Nk, 4))

for i in range(Nk):
    for j in range(Nk):
        H = make_H(KX[i,j], KY[i,j])
        eigs = np.linalg.eigvalsh(H)
        bands[i,j,:] = eigs

# Create interactive plotly surfaces
fig = go.Figure()

for n in range(4):
    fig.add_trace(go.Surface(
        x=KX, y=KY, z=bands[:,:,n],
        colorscale="Viridis",
        opacity=0.75,
        showscale=False,
        name=f"Band {n+1}"
    ))

fig.update_layout(
    title="Interactive 3D Bilayer Graphene Band Structure",
    scene=dict(
        xaxis_title="kx",
        yaxis_title="ky",
        zaxis_title="Energy (eV)"
    )
)

# Save to HTML
output_path = "./data/bilayer_graphene_3D_bands.html"
fig.write_html(output_path)

output_path
