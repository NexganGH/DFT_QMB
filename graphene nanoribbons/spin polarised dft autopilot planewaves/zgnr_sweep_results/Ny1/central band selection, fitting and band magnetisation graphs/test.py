import numpy as np
import matplotlib.pyplot as plt

# =======================================
# USER SETTINGS
# =======================================

npz_in  = "zgnr_Ny1_M1_bands_pi.npz"           # file DFT
npz_out = "zgnr_Ny1_M1_pi_two_above_from_Z.npz"  # file di output
spin_index = 0                                # 0 o 1 se spin–pol


# =======================================
# 1. Carico k ed energie E(k,n)
# =======================================

data = np.load(npz_in)

# k–asse (ka)
if "k_dimless" in data.files:
    k = data["k_dimless"]
elif "k" in data.files:
    k = data["k"]
else:
    raise KeyError(f"No k array in {npz_in}. Keys: {data.files}")

# energie (relative a EF)
E_all = None
for key in ["E_rel_all", "E_all", "E_rel", "E"]:
    if key in data.files:
        E_all = data[key]
        used_key = key
        break

if E_all is None:
    raise KeyError(f"No suitable energy array in {npz_in}. Keys: {data.files}")

print(f"Using energy key '{used_key}' from {npz_in}")

if E_all.ndim == 3:
    E = E_all[spin_index]    # (Nk, nbands)
elif E_all.ndim == 2:
    E = E_all
else:
    raise ValueError(f"Unexpected energy shape: {E_all.shape}")

Nk, nbands = E.shape
print(f"Loaded: Nk={Nk}, nbands={nbands}")

# =======================================
# 2. A Z scelgo le due bande SUBITO SOPRA EF
# =======================================

iZ = Nk - 1                 # ultimo k–punto = Z
Ek = E[iZ]

print("\nEnergies at Z (relative to EF):")
for n in range(nbands):
    print(f"  n={n:2d}, E_Z={Ek[n]:7.3f} eV")

mask_pos = Ek > 0.0
if np.sum(mask_pos) < 2:
    raise RuntimeError("Less than two bands above EF at Z.")

pos_idx = np.where(mask_pos)[0]
# ordina per energia crescente
pos_idx = pos_idx[np.argsort(Ek[pos_idx])]

n1 = pos_idx[0]   # più vicino sopra EF
n2 = pos_idx[1]   # seconda più vicina sopra EF

print(f"\nChosen two central bands at Z (both above EF):")
print(f"  n1 = {n1}, E_Z = {Ek[n1]:.3f} eV")
print(f"  n2 = {n2}, E_Z = {Ek[n2]:.3f} eV\n")


# =======================================
# 3. Tracking per continuità (globale, no limiti)
# =======================================

def track_band_by_continuity(E, i0, n0):
    """
    Segue una banda scegliendo a ogni k il livello
    con energia più vicina a quella precedente.
    Nessun limite sugli indici.
    """
    Nk, nbands = E.shape
    idx = np.empty(Nk, dtype=int)
    idx[i0] = n0

    # indietro in k: Z -> Γ
    for i in range(i0 - 1, -1, -1):
        E_prev = E[i + 1, idx[i + 1]]
        diffs = np.abs(E[i] - E_prev)
        idx[i] = np.argmin(diffs)

    # avanti in k (in pratica non serve, ma lo lascio per completezza)
    for i in range(i0 + 1, Nk):
        E_prev = E[i - 1, idx[i - 1]]
        diffs = np.abs(E[i] - E_prev)
        idx[i] = np.argmin(diffs)

    return idx


idx1 = track_band_by_continuity(E, iZ, n1)
idx2 = track_band_by_continuity(E, iZ, n2)

E1 = E[np.arange(Nk), idx1]   # banda “blu”
E2 = E[np.arange(Nk), idx2]   # banda “verde” che vuoi


# =======================================
# 4. Plot: tutte le bande + le due centrali
# =======================================

fig, ax = plt.subplots(figsize=(6, 5))

for n in range(nbands):
    ax.plot(k/np.pi, E[:, n], lw=0.6, color="0.8", zorder=1)

ax.plot(k/np.pi, E1, lw=2.5, color="C0", label="central band 1 (above EF)")
ax.plot(k/np.pi, E2, lw=2.5, color="C2", label="central band 2 (above EF)")

ax.axhline(0.0, ls="--", lw=0.8, color="k", alpha=0.6)
ax.set_xlabel(r"$k a/\pi$")
ax.set_ylabel(r"$E - E_F$ (eV)")
ax.set_title("Two central bands (both starting above EF at Z)")
ax.legend()
fig.tight_layout()
fig.savefig("central_two_above_from_Z.png", dpi=300)
print("Saved plot: central_two_above_from_Z.png")

# =======================================
# 5. Salvo i dati per il fit
# =======================================

np.savez(
    npz_out,
    k_dimless=k,
    E1=E1,
    E2=E2,
    idx1=idx1,
    idx2=idx2,
    n1_Z=n1,
    n2_Z=n2,
)

print(f"Saved central band data to: {npz_out}")
