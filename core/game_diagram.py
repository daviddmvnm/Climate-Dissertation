"""
game_diagram.py
───────────────
Three publication-quality diagrams for the dissertation model section.
Each formula is paired with a plain-English description above it.

  Diagram 1 — One-period flow ("the turn of the game")
  Diagram 2 — Irreversible adoption lock (circles, 3 panels)
  Diagram 3 — Weighted adoption W_t thermometer

Outputs
-------
  results/figures/fig_game_flow.png
  results/figures/fig_game_lock.png
  results/figures/fig_game_thermometer.png
  results/dissertation/fig_game_*.png   (copies)
"""

import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, FancyBboxPatch

matplotlib.rcParams.update({
    'mathtext.fontset': 'stix',
    'font.family':      'STIXGeneral',
    'font.size':        10,
})

os.makedirs(os.path.join(ROOT_DIR, "results", "figures"),      exist_ok=True)
os.makedirs(os.path.join(ROOT_DIR, "results", "dissertation"), exist_ok=True)

BLOC_COLORS = {
    'US':  '#1565C0',
    'EU':  '#2E7D32',
    'CN':  '#C62828',
    'RoW': '#E65100',
}
WEIGHTS = {'EU': 0.22, 'US': 0.28, 'CN': 0.30, 'RoW': 0.20}
THETA   = 0.80


# ── helpers ───────────────────────────────────────────────────────────────────

def rbox(ax, x, y, w, h, fc, ec, lw=1.5, rad=0.05, zorder=2):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={rad}",
                       facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder)
    ax.add_patch(p)

def arr(ax, x0, y0, x1, y1, col='#444', lw=1.6):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=col, lw=lw,
                                connectionstyle='arc3,rad=0.0'), zorder=6)

def eng(ax, x, y, s, ecol='#555', **kw):
    """Plain-English label — italic, grey, small."""
    kw.setdefault('ha', 'center'); kw.setdefault('va', 'center')
    ax.text(x, y, s, fontsize=8, color=ecol, style='italic', zorder=5, **kw)

def math(ax, x, y, s, col='#111', fs=9, **kw):
    """Math formula label."""
    kw.setdefault('ha', 'center'); kw.setdefault('va', 'center')
    ax.text(x, y, s, fontsize=fs, color=col, zorder=5, **kw)

def title_txt(ax, x, y, s, col, **kw):
    kw.setdefault('ha', 'center'); kw.setdefault('va', 'center')
    ax.text(x, y, s, fontsize=9.5, fontweight='bold', color=col, zorder=5, **kw)

def em(ax, x, y, etxt, mtxt, gap=0.30, ecol='#555', mcol='#111', mfs=9, **kw):
    """English line then maths line, centred on (x, y).
       English sits gap/2 above y, maths gap/2 below y."""
    eng(ax,  x, y + gap/2, etxt, ecol=ecol, **kw)
    math(ax, x, y - gap/2, mtxt, col=mcol, fs=mfs, **kw)


# ══════════════════════════════════════════════════════════════════════════════
# DIAGRAM 1 — ONE-PERIOD FLOW
# ══════════════════════════════════════════════════════════════════════════════

fig1, ax = plt.subplots(figsize=(16, 7.0))
ax.set_xlim(0, 16); ax.set_ylim(0, 7.0); ax.axis('off')

# colour palette  (face, edge)
SC = ('#E3F2FD', '#1565C0')
DC = ('#FFF8E1', '#E65100')
PC = ('#E8F5E9', '#1B5E20')
UC = ('#F3E5F5', '#4A148C')

# ── Box 1: State ─────────────────────────────────────────────────────────────
rbox(ax, 0.25, 1.30, 3.00, 4.80, *SC)
title_txt(ax, 1.75, 5.80, 'State at start of period $t$', SC[1])

em(ax, 1.75, 5.15,
   'Full game state (time + who adopted)',
   r'$s_t = (t,\; G_t)$', gap=0.32)

em(ax, 1.75, 4.45,
   "Each bloc's adoption status (0 = waiting, 1 = adopted)",
   r'$G_t = (G_{1t},\, G_{2t},\, G_{3t},\, G_{4t})$', gap=0.32)

em(ax, 1.75, 3.70,
   'Influence-weighted global adoption share',
   r'$W_t = \sum_i w_i\, G_{it}$', gap=0.32)

em(ax, 1.75, 2.95,
   'Set of blocs still deciding (not yet adopted)',
   r'$\mathcal{A}_t = \{i : G_{it} = 0\}$', gap=0.32)

em(ax, 1.75, 2.15,
   'Smooth coordination threshold (0 to 1)',
   r'$S(W_t) = \frac{1}{1+\exp(-\eta(W_t - \theta))}$', gap=0.32, mfs=8.5)

arr(ax, 3.25, 3.70, 3.70, 3.70)

# ── Box 2: Decision ──────────────────────────────────────────────────────────
rbox(ax, 3.70, 0.95, 3.40, 5.25, *DC)
title_txt(ax, 5.40, 5.85, 'Simultaneous decision', DC[1])

eng(ax, 5.40, 5.35, 'Every non-adopted bloc chooses at the same time')
em(ax, 5.40, 4.75,
   'Active blocs (those still deciding)',
   r'$i \in \mathcal{A}_t$', gap=0.30)
em(ax, 5.40, 4.10,
   'Binary action: adopt this period or wait',
   r'$a_{it} \in \{0,\, 1\}$', gap=0.30)

# Adopt sub-box
rbox(ax, 3.90, 2.65, 3.00, 0.95, '#C8E6C9', '#1B5E20', lw=1, rad=0.03)
eng(ax, 5.40, 3.32, 'Join the coalition this period', ecol='#1B5E20')
math(ax, 5.40, 2.98, r'Adopt: $a_{it} = 1$', col='#1B5E20', fs=8.5)

# Delay sub-box
rbox(ax, 3.90, 1.50, 3.00, 0.95, '#FFCDD2', '#B71C1C', lw=1, rad=0.03)
eng(ax, 5.40, 2.17, 'Hold out for another period', ecol='#B71C1C')
math(ax, 5.40, 1.83, r'Delay: $a_{it} = 0$', col='#B71C1C', fs=8.5)

arr(ax, 7.10, 3.70, 7.55, 3.70)

# ── Box 3: Payoffs ────────────────────────────────────────────────────────────
rbox(ax, 7.55, 0.50, 4.60, 5.90, *PC)
title_txt(ax, 9.85, 6.10, 'Period payoffs realised', PC[1])

# Adopters sub-box
rbox(ax, 7.75, 4.45, 4.20, 1.30, '#C8E6C9', '#1B5E20', lw=1, rad=0.03)
eng(ax, 9.85, 5.50, "Adopters pay a one-off transition cost", ecol='#1B5E20')
eng(ax, 9.85, 5.18, "(cost falls the more the world has already adopted)", ecol='#1B5E20')
math(ax, 9.85, 4.80, r'$-c_i^0\,(1 - \gamma W_t)$', col='#1B5E20', fs=9)

# Delayers sub-box
rbox(ax, 7.75, 2.95, 4.20, 1.25, '#FFCDD2', '#B71C1C', lw=1, rad=0.03)
eng(ax, 9.85, 3.95, 'Delayers face political pressure to join', ecol='#B71C1C')
eng(ax, 9.85, 3.63, '(pressure rises as the coalition grows)', ecol='#B71C1C')
math(ax, 9.85, 3.25, r'$-p_i\, W_t$', col='#B71C1C', fs=9)

# All players sub-box
rbox(ax, 7.75, 0.75, 4.20, 1.95, '#E0F2F1', '#004D40', lw=1, rad=0.03)
eng(ax, 9.85, 2.45, 'Everyone receives stabilisation benefits', ecol='#004D40')
eng(ax, 9.85, 2.13, 'and suffers climate damages — shared, non-excludable', ecol='#004D40')
math(ax, 9.85, 1.70, r'$-d_i^0(1+\kappa t)\,(1-S(W_t))$', col='#37474F', fs=8.5)
math(ax, 9.85, 1.28, r'$+\; b\cdot S(W_t)$', col='#37474F', fs=8.5)

arr(ax, 12.15, 3.70, 12.60, 3.70)

# ── Box 4: State update ───────────────────────────────────────────────────────
rbox(ax, 12.60, 1.30, 3.10, 4.30, *UC)
title_txt(ax, 14.15, 5.30, 'State update', UC[1])

em(ax, 14.15, 4.60,
   'Adoption is permanent — no going back',
   r'$G_{i,t+1} = G_{it} + (1-G_{it})\,a_{it}$', gap=0.32)

em(ax, 14.15, 3.70,
   'Once adopted, status is locked forever',
   r'$G_{it}=1 \;\Rightarrow\; G_{i,t+1}=1$', gap=0.32)

em(ax, 14.15, 2.75,
   'Next period uses updated adoption vector',
   r'$W_{t+1} = \sum_i w_i\, G_{i,t+1}$', gap=0.32)

em(ax, 14.15, 1.90,
   'Blocs who adopted leave the active set',
   r'$\mathcal{A}_{t+1} \subseteq \mathcal{A}_t$', gap=0.30)

# ── Loop-back arrow t → t+1 ───────────────────────────────────────────────────
ax.plot([15.70, 15.70], [1.30, 0.22], color='#666', lw=1.5, zorder=4)
ax.plot([15.70, 0.25],  [0.22, 0.22], color='#666', lw=1.5, zorder=4)
arr(ax, 0.25, 0.22, 0.25, 1.30, col='#666', lw=1.5)
math(ax, 8.00, 0.09, r'$t \;\longrightarrow\; t+1$  (repeat until $t = T$)',
     col='#666', fs=8.5)

fig1.suptitle('One-Period Flow of the Climate Cooperation Game',
              fontsize=13, fontweight='bold', y=0.99)
fig1.tight_layout(rect=[0, 0, 1, 0.97])

for p in [os.path.join(ROOT_DIR, 'results', 'figures', 'fig_game_flow.png'),
          os.path.join(ROOT_DIR, 'results', 'dissertation', 'fig_game_flow.png')]:
    fig1.savefig(p, dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig1)
print("Saved: fig_game_flow.png")


# ══════════════════════════════════════════════════════════════════════════════
# DIAGRAM 2 — IRREVERSIBLE ADOPTION LOCK
# ══════════════════════════════════════════════════════════════════════════════

BLOCS = ['US', 'EU', 'CN', 'RoW']

PANELS = [
    dict(title='Period $t = 1$\n(no adoption yet)',  adopted=set()),
    dict(title='Period $t = 3$\n(EU adopts)',         adopted={'EU'}),
    dict(title='Period $t = 5$\n(US joins)',          adopted={'EU', 'US'}),
]

fig2, axes2 = plt.subplots(1, 3, figsize=(12, 5.0))
fig2.suptitle('Irreversible Adoption — Active Set $\\mathcal{A}_t$ Shrinks Over Time',
              fontsize=12, fontweight='bold', y=1.01)

CX_POSITIONS = [0.55, 1.55, 2.55, 3.55]
R = 0.38

for ax, panel in zip(axes2, PANELS):
    adopted = panel['adopted']
    ax.set_xlim(0, 4.1); ax.set_ylim(-0.5, 4.0); ax.axis('off')
    ax.set_title(panel['title'], fontsize=10, fontweight='bold', pad=10)

    for j, bloc in enumerate(BLOCS):
        cx = CX_POSITIONS[j]
        cy = 2.35
        is_ad = bloc in adopted

        fc = BLOC_COLORS[bloc] if is_ad else 'white'
        ec = BLOC_COLORS[bloc]

        circ = Circle((cx, cy), R, facecolor=fc, edgecolor=ec,
                      linewidth=2.2, zorder=3)
        ax.add_patch(circ)

        tc = 'white' if is_ad else BLOC_COLORS[bloc]
        ax.text(cx, cy, bloc, ha='center', va='center',
                fontsize=9, fontweight='bold', color=tc, zorder=5)

        gv = '$G_{it}=1$' if is_ad else '$G_{it}=0$'
        ax.text(cx, cy - 0.60, gv, ha='center', va='top',
                fontsize=7.5, color='#555', zorder=5)

        if is_ad:
            ax.text(cx, cy - 1.05, 'adopted\n(locked)', ha='center', va='top',
                    fontsize=7, color=BLOC_COLORS[bloc], fontweight='bold', zorder=5)
            lx, ly, lw2, lh = cx - 0.13, cy + 0.30, 0.26, 0.16
            lock_box = FancyBboxPatch((lx, ly), lw2, lh,
                                      boxstyle="round,pad=0.01",
                                      fc='#FFF176', ec='#F9A825', lw=1.2, zorder=6)
            ax.add_patch(lock_box)
            ax.text(cx, ly + lh / 2, 'locked', ha='center', va='center',
                    fontsize=5.5, color='#795548', zorder=7)
        else:
            ax.text(cx, cy - 1.05, r'active', ha='center', va='top',
                    fontsize=7, color='#888', zorder=5)

    active_blocs = [b for b in BLOCS if b not in adopted]
    if active_blocs:
        act_str = r'$\mathcal{A}_t = \{$' + ', '.join(active_blocs) + r'$\}$'
    else:
        act_str = r'$\mathcal{A}_t = \emptyset$'
    ax.text(2.05, 0.65, act_str, ha='center', va='center',
            fontsize=9, color='#333',
            bbox=dict(boxstyle='round,pad=0.35', fc='#E8EAF6', ec='#3949AB', lw=1.2),
            zorder=5)

    W = sum(WEIGHTS[b] for b in adopted)
    ax.text(2.05, 0.05, f'$W_t = {W:.2f}$', ha='center', va='center',
            fontsize=9.5, fontweight='bold', color='#1565C0',
            bbox=dict(boxstyle='round,pad=0.3', fc='#E3F2FD', ec='#1565C0', lw=1.2),
            zorder=5)

axes2[1].annotate('Once $G_{it}=1$,\nno return',
                  xy=(CX_POSITIONS[BLOCS.index('EU')], 2.35 + R + 0.05),
                  xytext=(CX_POSITIONS[BLOCS.index('EU')], 3.55),
                  ha='center', fontsize=8, color='#B71C1C',
                  arrowprops=dict(arrowstyle='->', color='#B71C1C', lw=1.3),
                  bbox=dict(boxstyle='round,pad=0.25', fc='#FFEBEE', ec='#B71C1C', lw=1),
                  zorder=8)

fig2.tight_layout()
for p in [os.path.join(ROOT_DIR, 'results', 'figures', 'fig_game_lock.png'),
          os.path.join(ROOT_DIR, 'results', 'dissertation', 'fig_game_lock.png')]:
    fig2.savefig(p, dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig2)
print("Saved: fig_game_lock.png")


# ══════════════════════════════════════════════════════════════════════════════
# DIAGRAM 3 — WEIGHTED ADOPTION THERMOMETER
# ══════════════════════════════════════════════════════════════════════════════

EXAMPLES = [
    dict(label='EU only',      adopted=['EU']),
    dict(label='EU + US',      adopted=['EU', 'US']),
    dict(label='EU + US + CN', adopted=['EU', 'US', 'CN']),
]

STACK_ORDER = ['EU', 'US', 'CN', 'RoW']

fig3, axes3 = plt.subplots(1, 3, figsize=(10, 6.5), sharey=True)
fig3.suptitle(
    r'Weighted Adoption  $W_t = \sum_i w_i G_{it}$  and Coordination Threshold $\theta = 0.80$',
    fontsize=11, fontweight='bold', y=1.02)

BAR_X = 0.25; BAR_W = 0.50

for ax, ex in zip(axes3, EXAMPLES):
    adopted = ex['adopted']
    W = sum(WEIGHTS[b] for b in adopted)

    ax.set_xlim(0, 1); ax.set_ylim(-0.07, 1.13)
    ax.axis('off')
    ax.set_title(ex['label'], fontsize=10.5, fontweight='bold', pad=10)

    bg = FancyBboxPatch((BAR_X, 0), BAR_W, 1.0,
                        boxstyle="round,pad=0.005",
                        facecolor='#F5F5F5', edgecolor='#BDBDBD', lw=1.2, zorder=2)
    ax.add_patch(bg)

    bottom = 0.0
    for bloc in STACK_ORDER:
        h   = WEIGHTS[bloc]
        is_ad = bloc in adopted
        fc  = BLOC_COLORS[bloc] if is_ad else '#E0E0E0'
        ec  = BLOC_COLORS[bloc]
        alp = 1.0 if is_ad else 0.5

        seg = mpatches.Rectangle((BAR_X, bottom), BAR_W, h,
                                  facecolor=fc, edgecolor=ec,
                                  linewidth=1.3, alpha=alp, zorder=3)
        ax.add_patch(seg)

        mid = bottom + h / 2
        lc  = 'white' if is_ad else '#9E9E9E'
        fw  = 'bold' if is_ad else 'normal'
        ax.text(BAR_X + BAR_W / 2, mid,
                f'{bloc}\n$w={h:.2f}$',
                ha='center', va='center', fontsize=8, color=lc,
                fontweight=fw, zorder=5)

        bnd = bottom + h
        if bnd < 1.001:
            ax.plot([BAR_X + BAR_W, BAR_X + BAR_W + 0.06],
                    [bnd, bnd], color='#777', lw=0.8, zorder=4)
            ax.text(BAR_X + BAR_W + 0.08, bnd,
                    f'{bnd:.2f}', va='center', fontsize=7.5, color='#555')
        bottom += h

    ax.axhline(THETA, xmin=0.15, xmax=0.85,
               color='#C62828', lw=2.2, linestyle='--', zorder=7)
    ax.text(BAR_X + BAR_W + 0.08, THETA + 0.02,
            r'$\theta\!=\!0.80$', va='bottom', fontsize=9,
            color='#C62828', fontweight='bold')

    ax.annotate('', xy=(BAR_X - 0.04, W), xytext=(BAR_X - 0.18, W),
                arrowprops=dict(arrowstyle='->', color='#1A237E', lw=1.8), zorder=8)
    ax.text(BAR_X - 0.20, W, f'$W_t={W:.2f}$',
            ha='right', va='center', fontsize=9.5,
            color='#1A237E', fontweight='bold')

    met = W >= THETA - 1e-9
    vc  = '#2E7D32' if met else '#C62828'
    vt  = r'$W_t \geq \theta$  — threshold met' if met else r'$W_t < \theta$  — below threshold'
    ax.text(BAR_X + BAR_W / 2, -0.05, vt,
            ha='center', va='top', fontsize=8.5,
            color=vc, fontweight='bold', zorder=5)

ax0 = axes3[0]
ax0.axis('on')
ax0.set_xlim(0, 1); ax0.set_ylim(-0.07, 1.13)
ax0.set_yticks(np.arange(0, 1.01, 0.2))
ax0.set_yticklabels([f'{v:.1f}' for v in np.arange(0, 1.01, 0.2)], fontsize=8.5)
ax0.set_xticks([])
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.spines['bottom'].set_visible(False)
ax0.set_ylabel(r'$W_t$  (weighted adoption share)', fontsize=9.5, labelpad=6)

fig3.tight_layout()
for p in [os.path.join(ROOT_DIR, 'results', 'figures', 'fig_game_thermometer.png'),
          os.path.join(ROOT_DIR, 'results', 'dissertation', 'fig_game_thermometer.png')]:
    fig3.savefig(p, dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig3)
print("Saved: fig_game_thermometer.png")

print("\nAll three diagrams written.")
