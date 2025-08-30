#!/usr/bin/env python3
# strain_analysis.py
# Compute strain invariants, principal values/vectors, max shear, octahedral strains,
# von-Mises equivalent strain, Lode angle, Mohr-circle data, plot Mohr circles,
# and save a text report.

import numpy as np
import matplotlib.pyplot as plt

# ---------- core helpers ----------
def symmetrize(A, tol=1e-12):
    A = np.asarray(A, dtype=float)
    Asym = 0.5 * (A + A.T)
    if np.linalg.norm(A - A.T) > tol:
        print("[note] Input not perfectly symmetric; symmetrizing.")
    return Asym

def principal_invariants(E):
    I1 = np.trace(E)
    I2 = 0.5 * (I1**2 - np.trace(E @ E))
    I3 = np.linalg.det(E)
    return I1, I2, I3

def deviatoric_invariants(E):
    I1 = np.trace(E)
    s = E - (I1/3.0) * np.eye(3)
    J2p = 0.5 * np.sum(s*s)     # 0.5*s_ij*s_ij >= 0
    J3p = np.linalg.det(s)
    return J2p, J3p

def principal_strains_and_dirs(E):
    w, V = np.linalg.eigh(E)    # ascending: e1<=e2<=e3
    return w, V

def max_shear_strain(principal_strains):
    e1, _, e3 = principal_strains
    gamma_max = e3 - e1                # engineering max shear
    tau_max   = 0.5 * gamma_max        # tensor shear magnitude
    return gamma_max, tau_max

def octahedral_strains(principal_strains):
    e1, e2, e3 = principal_strains
    eps_oct  = (e1 + e2 + e3) / 3.0
    gamma_oct = (2.0/3.0) * np.sqrt((e1-e2)**2 + (e2-e3)**2 + (e3-e1)**2)
    return eps_oct, gamma_oct

# ---------- useful extras ----------
def mises_equiv_strain_from_dev(J2p):
    # J2p = 0.5*s:s
    return np.sqrt((4.0/3.0) * J2p)

def lode_angle(J2p, J3p):
    if J2p <= 0:
        return 0.0
    arg = (3.0*np.sqrt(3.0)/2.0) * (J3p / (J2p**1.5))
    arg = np.clip(arg, -1.0, 1.0)
    return (1.0/3.0) * np.arcsin(arg)  # radians

def rotate_tensor(E, Q):
    return Q @ E @ Q.T

def mohr_circles_from_principal(eigs):
    e1, e2, e3 = np.sort(eigs)
    centers = [(e1+e2)/2, (e2+e3)/2, (e3+e1)/2]
    radii   = [abs(e2-e1)/2, abs(e3-e2)/2, abs(e3-e1)/2]
    return centers, radii

def pretty_mu(x):
    return x * 1e6

# ---------- report text builder ----------
def format_report_text(res):
    e1, e2, e3 = res["principal_strains"]
    lines = []
    lines.append("=== Strain Analysis Report ===")
    lines.append("Input interpreted as absolute strain (dimensionless).")

    lines.append("\n[Principal invariants of E]")
    lines.append(f"I1 = tr(E)                = {res['I1']:.12e}  ({pretty_mu(res['I1']):.3f} μɛ)")
    lines.append(f"I2 = 0.5[(trE)^2-tr(E^2)] = {res['I2']:.12e}  ({res['I2']*1e12:.3f} (μɛ)^2)")
    lines.append(f"I3 = det(E)               = {res['I3']:.12e}  ({res['I3']*1e18:.3f} (μɛ)^3)")

    lines.append("\n[Deviatoric invariants (s = E - I1/3 I)]")
    lines.append(f"J2' = 0.5*s_ij*s_ij       = {res['J2_dev']:.12e}  ({res['J2_dev']*1e12:.3f} (μɛ)^2)")
    lines.append(f"J3' = det(s)              = {res['J3_dev']:.12e}  ({res['J3_dev']*1e18:.3f} (μɛ)^3)")

    lines.append("\n[Principal strains and directions]")
    lines.append(f"e1 = {e1:.12e}  ({pretty_mu(e1):.3f} μɛ)")
    lines.append(f"e2 = {e2:.12e}  ({pretty_mu(e2):.3f} μɛ)")
    lines.append(f"e3 = {e3:.12e}  ({pretty_mu(e3):.3f} μɛ)")
    lines.append("Eigenvectors (columns, unit):")
    with np.printoptions(precision=6, suppress=True):
        lines.append(str(res["principal_directions"]))

    lines.append("\n[Maximum shear]")
    lines.append(f"gamma_max (engineering) = e3 - e1 = {res['gamma_max_engineering']:.12e}  ({pretty_mu(res['gamma_max_engineering']):.3f} μɛ)")
    lines.append(f"tau_max (tensor)        = 0.5*gamma_max = {res['tau_max_tensor']:.12e}  ({pretty_mu(res['tau_max_tensor']):.3f} μɛ)")

    lines.append("\n[Octahedral strains]")
    lines.append(f"eps_oct  = (e1+e2+e3)/3 = {res['eps_octahedral']:.12e}  ({pretty_mu(res['eps_octahedral']):.3f} μɛ)")
    lines.append(f"gamma_oct (engineering) = {res['gamma_octahedral']:.12e}  ({pretty_mu(res['gamma_octahedral']):.3f} μɛ)")
    lines.append("Check: gamma_oct = sqrt(8/3 * J2') -> " +
                 f"{np.sqrt(8.0/3.0 * res['J2_dev']):.12e}")

    lines.append("\n[Additional descriptors]")
    lines.append(f"von Mises equivalent strain = {res['vm_eq']:.12e}  ({pretty_mu(res['vm_eq']):.3f} μɛ)")
    lines.append(f"Lode angle theta            = {res['lode_theta']:.6f} rad")
    centers, radii = res["mohr_centers"], res["mohr_radii"]
    lines.append("Mohr circles (centers, radii):")
    for k in range(3):
        lines.append(f"  circle {k+1}: c={centers[k]:.12e}, r={radii[k]:.12e}")
    return "\n".join(lines)

def save_report_text(res, path="strain_report.txt"):
    txt = format_report_text(res)
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)
    print(f"[saved] report -> {path}")

# ---------- plotting ----------
def plot_mohr_circles(res, outfile="mohr_strain.png",
                      show=False,
                      use_microstrain_units=True,
                      ord_is_engineering_shear=False):
    """
    Plot Mohr circles for strain.
    x-axis: normal strain ε
    y-axis: tensor shear ε_ij (default) or engineering shear γ_ij (if ord_is_engineering_shear=True)

    Notes:
      - With principal strains e1<=e2<=e3, circles are:
           (c12, r12), (c23, r23), (c31, r31)
      - r31 is the largest; gamma_max = 2*r31 (engineering shear)
    """
    e1, e2, e3 = res["principal_strains"]
    centers, radii = res["mohr_centers"], res["mohr_radii"]

    # scale and labels
    scale = 1e6 if use_microstrain_units else 1.0
    unit = "μɛ" if use_microstrain_units else ""
    y_label = "engineering shear γ" if ord_is_engineering_shear else "tensor shear ε_ij"
    y_scale = 2.0*scale if ord_is_engineering_shear else scale  # γ = 2ε_ij

    # parametric circle
    th = np.linspace(0, 2*np.pi, 400)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for (c, r) in zip(centers, radii):
        x = (c + r*np.cos(th)) * scale
        y = (r*np.sin(th)) * y_scale
        ax.plot(x, y, linewidth=1.5)

    # principal points
    ax.plot([e1*scale, e2*scale, e3*scale], [0,0,0], marker='o', linestyle='none', label="principal points")

    # helpful guides
    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)
    ax.set_aspect('equal', adjustable='datalim')

    # annotate largest circle (r31)
    r31 = radii[2]
    c31 = centers[2]
    ax.annotate(f"r31 = {r31*scale:.1f} {unit}\n(γ_max = {2*r31*scale:.1f} {unit})",
                xy=((c31+r31)*scale, 0),
                xytext=(10, 10), textcoords='offset points')

    ax.set_xlabel(f"normal strain ε ({unit})")
    ax.set_ylabel(f"{y_label} ({unit})")
    ax.set_title("Mohr Circles for Strain")
    ax.legend(loc="best")

    plt.tight_layout()
    fig.savefig(outfile, dpi=200)
    print(f"[saved] Mohr circle figure -> {outfile}")
    if show:
        plt.show()

# ---------- main analysis ----------
def run_analysis(E, assume_microstrain=False):
    """
    E: 3x3 small-strain tensor. If assume_microstrain=True, values are μɛ.
    Off-diagonals are tensor shear strains ε_ij (engineering γ_ij = 2ε_ij).
    """
    E = np.array(E, dtype=float)
    if assume_microstrain:
        E = E * 1e-6

    E = symmetrize(E)

    I1, I2, I3       = principal_invariants(E)
    J2p, J3p         = deviatoric_invariants(E)
    eigvals, eigvecs = principal_strains_and_dirs(E)
    gamma_max, tau_max = max_shear_strain(eigvals)
    eps_oct, gamma_oct = octahedral_strains(eigvals)

    vm_eq  = mises_equiv_strain_from_dev(J2p)
    theta  = lode_angle(J2p, J3p)
    centers, radii = mohr_circles_from_principal(eigvals)

    return {
        "E": E,
        "I1": I1, "I2": I2, "I3": I3,
        "J2_dev": J2p, "J3_dev": J3p,
        "principal_strains": eigvals,
        "principal_directions": eigvecs,  # columns are eigenvectors (unit)
        "gamma_max_engineering": gamma_max,
        "tau_max_tensor": tau_max,
        "eps_octahedral": eps_oct,
        "gamma_octahedral": gamma_oct,
        "vm_eq": vm_eq,
        "lode_theta": theta,
        "mohr_centers": centers,
        "mohr_radii": radii,
    }

def print_report(res):
    txt = format_report_text(res)
    print(txt)

# ---------- example ----------
if __name__ == "__main__":
    # Your matrix (in microstrain)
    E_micro = np.array([
        [100,  -30,  -60],
        [-30, -300,   50],
        [-60,   50,  200]
    ], dtype=float)

    res = run_analysis(E_micro, assume_microstrain=True)

    # print to console
    print_report(res)

    # save text report
    save_report_text(res, path="strain_report.txt")

    # plot and save Mohr circles (strain)
    plot_mohr_circles(
        res,
        outfile="mohr_strain.png",
        show=False,                    # set True if you want a window to pop up
        use_microstrain_units=True,    # label axes in μɛ
        ord_is_engineering_shear=False # y-axis = tensor shear ε_ij; set True for engineering γ
    )
