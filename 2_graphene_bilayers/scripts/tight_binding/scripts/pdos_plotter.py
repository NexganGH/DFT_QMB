# pdos_graphene.py
import os
import numpy as np
import matplotlib.pyplot as plt
from gpaw.dos import DOSCalculator
import plotly.graph_objects as go


def plot_pdos_pi_sigma(calc,
                       atoms,
                       atom_indices=None,
                       emin=-10.0,
                       emax=10.0,
                       npts=2001,
                       width=0.2,
                       zoom=(-5, 5),
                       save_html=True,
                       html_path="../data/graphene_pdos_interactive.html"):
    """
    Compute PDOS π (p_z) and σ (p_x+p_y), plot with matplotlib,
    and save an interactive Plotly HTML figure.

    Parameters match your old script.
    """

    # --------------------------
    # Default atoms to average over
    # --------------------------
    if atom_indices is None:
        atom_indices = list(range(len(atoms)))

    # --------------------------
    # DOS calculator
    # --------------------------
    doscalc = DOSCalculator.from_calculator(calc)
    energies = np.linspace(emin, emax, npts)

    # --------------------------
    # Helper: average PDOS over atoms
    # --------------------------
    def avg_pdos(l, m=None):
        pdos_sum = np.zeros_like(energies)
        for a in atom_indices:
            pdos = doscalc.raw_pdos(
                energies,
                a=a,
                l=l,
                m=m,
                spin=None,
                width=width
            )
            pdos_sum += pdos
        return pdos_sum / len(atom_indices)

    # --------------------------
    # Compute channels
    # --------------------------
    pdos_s = avg_pdos(l=0)
    pdos_p_total = avg_pdos(l=1)
    pdos_pz = avg_pdos(l=1, m=1)             # π = p_z
    pdos_sigma = pdos_p_total - pdos_pz      # σ = p_x + p_y

    # Clip numerical noise
    pdos_pz = np.clip(pdos_pz, 0, None)
    pdos_sigma = np.clip(pdos_sigma, 0, None)

    # ===========================================================
    # 1) Matplotlib static plot
    # ===========================================================
    plt.figure(figsize=(7, 5))
    plt.plot(energies, pdos_pz, label=r'$\pi$ (p$_z$)', lw=2)
    plt.plot(energies, pdos_sigma, label=r'$\sigma$ (p$_x$+p$_y$)', lw=2, alpha=0.7)
    plt.plot(energies, pdos_s, label='s', lw=1, alpha=0.5)

    plt.xlim(*zoom)
    plt.xlabel('Energy − $E_F$ (eV)')
    plt.ylabel('PDOS (arb. units)')
    plt.title("Graphene PDOS: π and σ decomposition")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ===========================================================
    # 2) Interactive Plotly figure
    # ===========================================================
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=energies, y=pdos_pz,
        mode='lines', name='π (p_z)',
        line=dict(width=3)
    ))

    fig.add_trace(go.Scatter(
        x=energies, y=pdos_sigma,
        mode='lines', name='σ (p_x + p_y)',
        line=dict(width=3)
    ))

    fig.add_trace(go.Scatter(
        x=energies, y=pdos_s,
        mode='lines', name='s',
        line=dict(width=2)
    ))

    fig.update_layout(
        title="Interactive Graphene PDOS (π and σ)",
        xaxis_title="Energy − E_F (eV)",
        yaxis_title="PDOS (arb. units)",
        hovermode='x unified',
        template='plotly_white'
    )
    fig.update_xaxes(range=zoom)

    # Save HTML interactive file
    if save_html:
        os.makedirs(os.path.dirname(html_path), exist_ok=True)
        fig.write_html(html_path)
        print(f"Saved interactive PDOS to: {html_path}")

    return fig
