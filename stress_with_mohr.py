#!/usr/bin/env python3
# stress_analysis.py
# Compute stress invariants, principal values/vectors, max shear, octahedral stresses,
# von-Mises equivalent stress, Lode angle, Mohr circles, plot, and save report.
# input S is a stress tensor (MPa, say).
# off diag terms are symmetric
# Output includes invariants, principal stresses/directions, von Mises, Lode angle, Mohr’s circles, etc.
# Creates stress_report.txt and mohr_stress.png.


import numpy as np
import matplotlib.pyplot as plt

# ---------- core helpers ----------
def symmetrize(A, tol=1e-12):
    A = np.asarray(A, dtype=float)
    Asym = 0.5 * (A + A.T)
    if np.linalg.norm(A - A.T) > tol:
        print("[note] Input not perfectly symmetric; symmetrizing.")
    return Asym

def principal_invariants(S):
    I1 = np.trace(S)
    I2 = 0.5 * (I1**2 - np.trace(S @ S))
    I3 = np.linalg.det(S)
    return I1, I2, I3

def deviatoric_invariants(S):
    I1 = np.trace(S)
    s = S - (I1/3.0) * np.eye(3)
    J2 = 0.5 * np.sum(s*s)     # 0.5*s_ij*s_ij >= 0
    J3 = np.linalg.det(s)
    return J2, J3, s

def principal_stresses_and_dirs(S):
    w, V = np.linalg.eigh(S)    # ascending: σ1<=σ2<=σ3
    return w, V

def max_shear_stress(principal_stresses):
    s1, _, s3 = principal_stresses
    tau_max = 0.5 * (s3 - s1)
    return tau_max

def octahedral_stresses(principal_stresses):
    s1, s2, s3 = principal_stresses
    sigma_oct = (s1 + s2 + s3) / 3.0
    tau_oct   = (1.0/3.0) * np.sqrt((s1-s2)**2 + (s2-s3)**2 + (s3-s1)**2)
    return sigma_oct, tau_oct

# ---------- extras ----------
def mises_eq_stress(J2):
    return np.sqrt(3.0 * J2)

def lode_angle(J2, J3):
    if J2 <= 0:
        return 0.0
    arg = (3.0*np.sqrt(3.0)/2.0) * (J3 / (J2**1.5))
    arg = np.clip(arg, -1.0, 1.0)
    return (1.0/3.0) * np.arcsin(arg)  # radians

def mohr_circles_from_principal(sigmas):
    s1, s2, s3 = np.sort(sigmas)
    centers = [(s1+s2)/2, (s2+s3)/2, (s3+s1)/2]
    radii   = [abs(s2-s1)/2, abs(s3-s2)/2, abs(s3-s1)/2]
    return centers, radii

# ---------- reporting ----------
def format_report_text(res):
    s1, s2, s3 = res["principal_stresses"]
    lines = []
    lines.append("=== Stress Analysis Report ===")

    lines.append("\n[Principal invariants of σ]")
    lines.append(f"I1 = {res['I1']:.6e}")
    lines.append(f"I2 = {res['I2']:.6e}")
    lines.append(f"I3 = {res['I3']:.6e}")

    lines.append("\n[Deviatoric invariants (s = σ - I1/3 I)]")
    lines.append(f"J2 = {res['J2']:.6e}")
    lines.append(f"J3 = {res['J3']:.6e}")

    lines.append("\n[Principal stresses and directions]")
    lines.append(f"σ1 = {s1:.6e}")
    lines.append(f"σ2 = {s2:.6e}")
    lines.append(f"σ3 = {s3:.6e}")
    with np.printoptions(precision=6, suppress=True):
        lines.append(str(res["principal_directions"]))

    lines.append("\n[Maximum shear stress]")
    lines.append(f"τ_max = {res['tau_max']:.6e}")

    lines.append("\n[Octahedral stresses]")
    lines.append(f"σ_oct  = {res['sigma_oct']:.6e}")
    lines.append(f"τ_oct  = {res['tau_oct']:.6e}")

    lines.append("\n[Additional descriptors]")
    lines.append(f"von Mises equivalent stress = {res['vm_eq']:.6e}")
    lines.append(f"Lode angle θ                = {res['lode_theta']:.6f} rad")
    centers, radii = res["mohr_centers"], res["mohr_radii"]
    lines.append("Mohr circles (centers, radii):")
    for k in range(3):
        lines.append(f"  circle {k+1}: c={centers[k]:.6e}, r={radii[k]:.6e}")
    return "\n".join(lines)

def save_report_text(res, path="stress_report.txt"):
    txt = format_report_text(res)
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)
    print(f"[saved] report -> {path}")

# ---------- plotting ----------
def plot_mohr_circles(res, outfile="mohr_stress.png", show=False):
    s1, s2, s3 = res["principal_stresses"]
    centers, radii = res["mohr_centers"], res["mohr_radii"]

    th = np.linspace(0, 2*np.pi, 400)
    fig, ax = plt.subplots()

    for (c, r) in zip(centers, radii):
        x = c + r*np.cos(th)
        y = r*np.sin(th)
        ax.plot(x, y, linewidth=1.5)

    # principal points
    ax.plot([s1, s2, s3], [0,0,0], 'o', label="principal stresses")

    # axes
    ax.axhline(0, color='k', linewidth=1)
    ax.axvline(0, color='k', linewidth=1)

    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel("Normal stress σ")
    ax.set_ylabel("Shear stress τ")
    ax.set_title("Mohr Circles for Stress")
    ax.legend(loc="best")

    plt.tight_layout()
    fig.savefig(outfile, dpi=200)
    print(f"[saved] Mohr circle figure -> {outfile}")
    if show:
        plt.show()

# ---------- main analysis ----------
def run_analysis(S):
    S = symmetrize(S)

    I1, I2, I3         = principal_invariants(S)
    J2, J3, devS       = deviatoric_invariants(S)
    eigvals, eigvecs   = principal_stresses_and_dirs(S)
    tau_max            = max_shear_stress(eigvals)
    sigma_oct, tau_oct = octahedral_stresses(eigvals)
    vm_eq              = mises_eq_stress(J2)
    theta              = lode_angle(J2, J3)
    centers, radii     = mohr_circles_from_principal(eigvals)

    return {
        "S": S,
        "I1": I1, "I2": I2, "I3": I3,
        "J2": J2, "J3": J3,
        "principal_stresses": eigvals,
        "principal_directions": eigvecs,
        "tau_max": tau_max,
        "sigma_oct": sigma_oct,
        "tau_oct": tau_oct,
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
    # Example stress tensor (MPa)
    S = np.array([
        [120,  40,   0],
        [ 40,  80, -20],
        [  0, -20,  60]
    ], dtype=float)

    res = run_analysis(S)

    # console report
    print_report(res)

    # save text report
    save_report_text(res, path="stress_report.txt")

    # plot Mohr circles
    plot_mohr_circles(res, outfile="mohr_stress.png", show=False)


