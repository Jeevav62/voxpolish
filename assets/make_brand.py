"""Generate VoxPolish brand assets: social banner + square logo."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import numpy as np

BG     = "#0f1117"
PANEL  = "#1a1d27"
INDIGO = "#6c63ff"
TEAL   = "#00d4aa"
RED    = "#ef4444"
GREEN  = "#22c55e"
WHITE  = "#f4f6fb"
MUTED  = "#8892a4"


def waveform(ax, x0, x1, y, noisy_frac=0.5, height=0.5, seed=7):
    """Draw a noisy->clean waveform from x0 to x1 centered on y."""
    rng = np.random.default_rng(seed)
    n = 220
    xs = np.linspace(x0, x1, n)
    split = x0 + (x1 - x0) * noisy_frac
    for i, x in enumerate(xs):
        if x < split:
            # noisy: dense, jittery, smaller, red
            amp = height * (0.25 + 0.35 * rng.random())
            color = RED
            alpha = 0.85
        else:
            # clean: speech-like bursts, green
            t = (x - split) / (x1 - split + 1e-9)
            env = abs(np.sin(t * np.pi * 3.2)) ** 1.3
            amp = height * (0.08 + 0.95 * env)
            color = GREEN
            alpha = 0.95
        ax.plot([x, x], [y - amp, y + amp], color=color, lw=1.6, alpha=alpha, solid_capstyle="round")


def mic(ax, cx, cy, s):
    """Draw a simple microphone glyph."""
    # capsule
    ax.add_patch(FancyBboxPatch((cx - 0.18*s, cy - 0.28*s), 0.36*s, 0.62*s,
                 boxstyle="round,pad=0,rounding_size=" + str(0.18*s),
                 fc=INDIGO, ec="none", zorder=5))
    # grille lines
    for k in range(3):
        yy = cy - 0.08*s + k*0.13*s
        ax.plot([cx-0.12*s, cx+0.12*s], [yy, yy], color=BG, lw=max(1, s*0.9), zorder=6)
    # stand arc
    th = np.linspace(np.pi*0.15, np.pi*0.85, 60)
    r = 0.30*s
    ax.plot(cx + r*np.cos(th), cy - 0.36*s - r*0.0 + r*np.sin(th)*-1 + r, color=TEAL, lw=max(2, s*1.4), zorder=4)
    ax.plot(cx + r*np.cos(th), (cy - 0.10*s) - r*np.sin(th), color=TEAL, lw=max(2, s*1.4), zorder=4)
    # post + base
    ax.plot([cx, cx], [cy - 0.40*s, cy - 0.62*s], color=TEAL, lw=max(2, s*1.4), zorder=4)
    ax.plot([cx-0.16*s, cx+0.16*s], [cy - 0.62*s, cy - 0.62*s], color=TEAL, lw=max(2, s*1.4), zorder=4)


# ── Banner 1280x640 ──────────────────────────────────────────────────────────
def make_banner():
    fig = plt.figure(figsize=(12.8, 6.4), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 12.8); ax.set_ylim(0, 6.4)
    ax.add_patch(Rectangle((0, 0), 12.8, 6.4, fc=BG, ec="none"))
    # subtle top glow bar
    ax.add_patch(Rectangle((0, 6.25), 12.8, 0.15, fc=INDIGO, ec="none", alpha=0.9))

    mic(ax, 2.0, 4.15, 1.5)

    ax.text(3.25, 4.35, "VoxPolish", color=WHITE, fontsize=72, fontweight="bold",
            va="center", ha="left", family="DejaVu Sans")
    ax.text(3.30, 3.35, "TTS Dataset Cleaner", color=INDIGO, fontsize=30,
            fontweight="bold", va="center", ha="left")

    # waveform strip
    waveform(ax, 1.2, 11.6, 1.7, noisy_frac=0.5, height=0.62, seed=11)
    ax.text(1.2, 0.78, "denoise", color=RED, fontsize=20, fontweight="bold", va="center", ha="left")
    ax.text(6.4, 0.78, "trim  ·  normalize", color=MUTED, fontsize=17, va="center", ha="center")
    ax.text(11.6, 0.78, "clean", color=GREEN, fontsize=20, fontweight="bold", va="center", ha="right")

    fig.savefig("banner.png", facecolor=BG)
    plt.close(fig)
    print("saved banner.png (1280x640)")


# ── Square logo 512x512 ──────────────────────────────────────────────────────
def make_logo():
    fig = plt.figure(figsize=(5.12, 5.12), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 5.12); ax.set_ylim(0, 5.12)
    # rounded panel bg
    ax.add_patch(Rectangle((0, 0), 5.12, 5.12, fc=BG, ec="none"))
    ax.add_patch(FancyBboxPatch((0.5, 0.5), 4.12, 4.12,
                 boxstyle="round,pad=0,rounding_size=0.7", fc=PANEL, ec=INDIGO, lw=3))
    mic(ax, 2.56, 3.15, 1.7)
    waveform(ax, 1.1, 4.0, 1.55, noisy_frac=0.5, height=0.45, seed=5)
    fig.savefig("logo.png", facecolor=BG)
    plt.close(fig)
    print("saved logo.png (512x512)")


make_banner()
make_logo()
