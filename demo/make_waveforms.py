"""Generate before/after waveform comparison PNGs for the README."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np

PAIRS = [
    ("jeeva_before.ogg", "jeeva_after.wav", "Voice Recording", "waveform_jeeva.png"),
    ("car_before.wav",   "car_after.wav",   "Background / Traffic Noise", "waveform_car.png"),
]

BG     = "#0f1117"
NOISY  = "#ef4444"
CLEAN  = "#22c55e"
TEXT   = "#e2e8f0"
MUTED  = "#8892a4"


def load_mono(path):
    d, r = sf.read(path)
    if d.ndim > 1:
        d = d.mean(axis=1)
    return d.astype(np.float32), r


for before, after, title, out in PAIRS:
    yb, rb = load_mono(before)
    ya, ra = load_mono(after)
    tb = np.linspace(0, len(yb) / rb, len(yb))
    ta = np.linspace(0, len(ya) / ra, len(ya))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 4.5), facecolor=BG)
    fig.suptitle(f"{title} — Before vs After", color=TEXT, fontsize=14, fontweight="bold")

    ax1.plot(tb, yb, color=NOISY, linewidth=0.4)
    ax1.set_title("BEFORE — noisy", color=NOISY, fontsize=11, loc="left", fontweight="bold")

    ax2.plot(ta, ya, color=CLEAN, linewidth=0.4)
    ax2.set_title("AFTER — cleaned (denoise + trim + normalize)", color=CLEAN, fontsize=11, loc="left", fontweight="bold")

    for ax in (ax1, ax2):
        ax.set_facecolor(BG)
        ax.set_ylim(-1.05, 1.05)
        ax.tick_params(colors=MUTED, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#2d3348")
        ax.set_xlabel("seconds", color=MUTED, fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out, dpi=120, facecolor=BG)
    plt.close()
    print(f"saved {out}")
