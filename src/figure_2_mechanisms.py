import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from pathlib import Path

plt.rcParams.update({'font.size': 12, 'mathtext.fontset': 'dejavusans'})

fig, ax = plt.subplots(figsize=(16, 10), facecolor='white')
fig.subplots_adjust(left=0.035, right=0.965, top=0.97, bottom=0.04)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')


def rounded_box(x, y, w, h, text, fontsize=12, weight='normal', lw=1.6, rounding=0.02, fc='white'):
    patch = FancyBboxPatch((x, y), w, h,
                           boxstyle=f'round,pad=0.012,rounding_size={rounding}',
                           linewidth=lw, edgecolor='black', facecolor=fc)
    ax.add_patch(patch)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, weight=weight, linespacing=1.15)
    return patch


def arrow(x1, y1, x2, y2, lw=1.45, ms=15):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                                 mutation_scale=ms, linewidth=lw, color='black',
                                 shrinkA=2, shrinkB=2))

# Title area with generous spacing
ax.text(0.5, 0.975, 'Mechanisms for handling the first post-interface BDF2 step',
        ha='center', va='center', fontsize=24, weight='bold')
ax.text(0.5, 0.915,
        'Raw crossing, restart, and interface-aware correction act at different stages of the update',
        ha='center', va='center', fontsize=16)

rounded_box(0.285, 0.815, 0.43, 0.075,
            r'Common starting point:' + '\n' + r'first post-interface BDF2 stencil crosses $t^\star$',
            fontsize=17, weight='bold', lw=1.8, fc='#fcfcfc')

# Geometry strip
xkm1, xk, xt, xkp1 = 0.31, 0.43, 0.56, 0.69
y_line = 0.735
ax.plot([0.26, 0.74], [y_line, y_line], color='black', linewidth=1.2)
for x in [xkm1, xk, xkp1]:
    ax.plot(x, y_line, 'o', color='black', ms=7)
ax.plot([xt, xt], [0.685, 0.785], '--', color='black', linewidth=1.1)
ax.text(0.365, 0.765, 'left jet', ha='center', fontsize=14)
ax.text(0.66, 0.765, 'right jet', ha='center', fontsize=14)
ax.text(xt, 0.785, r'$t^{\star}$', ha='center', va='bottom', fontsize=14)
ax.text(xkm1, 0.705, r'$t_{k-1}$', ha='center', va='top', fontsize=14)
ax.text(xk,   0.705, r'$t_k$', ha='center', va='top', fontsize=14)
ax.text(xkp1, 0.705, r'$t_{k+1}$', ha='center', va='top', fontsize=14)
ax.text(0.5, 0.655,
        'Same crossing configuration in all cases; the leading defect depends on J1, J2, and theta.',
        ha='center', va='center', fontsize=14.5)

# Columns
col_centers = [0.17, 0.50, 0.83]
for xc, hdr in zip(col_centers, ['STANDARD BDF2', 'RESTART', 'DIRECT CORRECTION']):
    ax.text(xc, 0.565, hdr, ha='center', va='center', fontsize=21, weight='bold')
for x in [0.335, 0.665]:
    ax.add_line(Line2D([x, x], [0.17, 0.54], color='0.6', linestyle=':', linewidth=1.0))

w = 0.24
xL, xM, xR = 0.05, 0.38, 0.71
y_top, y_mid, y_bot = 0.44, 0.30, 0.13
h_top, h_mid, h_bot = 0.065, 0.080, 0.108

# Left
rounded_box(xL, y_top, w, h_top,
            'Crossing value at $t_{k+1}$ is produced\nby the standard BDF2 stencil', fontsize=12.8)
rounded_box(xL, y_mid, w, h_mid,
            'First crossing residual keeps the\njump-generated mixed-side defect', fontsize=12.8)
rounded_box(xL, y_bot, w, h_bot,
            'Leading local scale:\nsevere regime: $O(1)$ generically\nbenign regime: $O(h)$', fontsize=12.8)

# Middle
rounded_box(xM, y_top, w, h_top,
            'Crossing value at $t_{k+1}$ is still produced\nby the same standard BDF2 stencil', fontsize=12.4)
rounded_box(xM, y_mid, w, h_mid,
            'Restart acts only after crossing:\nreinitialize post-interface history', fontsize=12.8)
rounded_box(xM, y_bot, w, h_bot,
            'Effect on first crossing residual:\nnone\n$R^{\\mathrm{restart}}_{k+1}=R^{\\mathrm{standard}}_{k+1}$', fontsize=12.3)

# Right
rounded_box(xR, y_top, w, h_top,
            'Modify the crossing update itself\nby subtracting the defect contribution', fontsize=12.4)
rounded_box(xR, y_mid, w, h_mid,
            'Target the jump terms directly:\n$c_1(\\theta)J_1 + h\\,c_2(\\theta)J_2$', fontsize=12.8)
rounded_box(xR, y_bot, w, h_bot,
            'Ideal local effect:\nremove the leading crossing obstruction\nrestore the residual to $O(h^2)$', fontsize=12.3)

for xc in col_centers:
    arrow(xc, y_top, xc, y_mid + h_mid, lw=1.4, ms=15)
    arrow(xc, y_mid, xc, y_bot + h_bot, lw=1.4, ms=15)

rounded_box(0.24, 0.02, 0.52, 0.065,
            'Key message: restart is a history-management mechanism,\nwhereas correction is a crossing-defect cancellation mechanism.',
            fontsize=15.8, weight='bold', lw=1.9, fc='#fcfcfc')

OUTDIR = Path(__file__).resolve().parents[1] / 'figures'
OUTDIR.mkdir(exist_ok=True)
plt.savefig(OUTDIR / 'figure_2_mechanisms.png', dpi=600)
plt.savefig(OUTDIR / 'figure_2_mechanisms.pdf')
